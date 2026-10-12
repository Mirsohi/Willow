"""Round 1's compute per stage, from the queue state and the decode summaries (read-only).

scout, 2026-10-12. For methods §4.14 of the Willow paper.

Inputs (decoder's files, read only):
  results/decoder/willow/queue_state.json        one record per job: wall seconds of its last attempt
  results/decoder/willow/decode/<leaf>/<tag>/summary.json
                                                  per decoder: n and ms_per_shot (CPU, time.process_time,
                                                  accumulated over every chunk, including chunks of earlier
                                                  attempts that a resumed job loaded)
Output: results/scout/willow_compute_round1.json

Two measures per stage (both measured, neither exact):
  queue_wall_h   sum over jobs of the last attempt's wall time. Each job ran one thread, so this bounds its CPU
                 from above for that attempt; it omits earlier, interrupted attempts.
  decoder_cpu_h  sum over decoders of n * ms_per_shot: CPU of the decoders on the selected shots only (no
                 syndrome reference samples, loading or checks); a lower bound on the stage's CPU.
Fit jobs have no decode summary; only their wall time is summed.
"""
import collections, hashlib, json, os, sys, time

ROOT = '/mnt/project-files/tensor-networks/deep-dives/experimental/'
QS = ROOT + 'results/decoder/willow/queue_state.json'
OUT = sys.argv[1] if len(sys.argv) > 1 else ROOT + 'results/scout/willow_compute_round1.json'


def md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


t0 = time.time()
raw = open(QS, 'rb').read()
q = json.loads(raw)
stages = collections.OrderedDict((s, {'jobs': 0, 'attempts': 0, 'queue_wall_h': 0.0, 'decoder_cpu_h': 0.0,
                                      'summaries_read': 0, 'first_start': None, 'last_end': None,
                                      'by_tag': {}}) for s in q['_stages'])
digest = hashlib.md5()
missing = []
for key in sorted(q['jobs']):
    rec = q['jobs'][key]
    st, leaf, tag = key.split('|')[:3]
    s = stages[st]
    s['jobs'] += 1
    s['attempts'] += rec['attempts']
    s['queue_wall_h'] += rec['seconds'] / 3600.0
    s['first_start'] = min(filter(None, [s['first_start'], rec['start']]))
    s['last_end'] = max(filter(None, [s['last_end'], rec['end']]))
    t = s['by_tag'].setdefault(tag, {'jobs': 0, 'queue_wall_h': 0.0, 'decoder_cpu_h': 0.0})
    t['jobs'] += 1
    t['queue_wall_h'] += rec['seconds'] / 3600.0
    if st == 'fit':
        continue
    fn = os.path.join(ROOT, 'results/decoder/willow/decode', leaf, tag, 'summary.json')
    try:
        b = open(fn, 'rb').read()
    except OSError:
        missing.append(key)
        continue
    digest.update(hashlib.md5(b).hexdigest().encode())
    summ = json.loads(b)
    cpu = sum(d['ms_per_shot'] * d['n'] / 1000.0 for d in summ.get('decoders', {}).values()
              if isinstance(d, dict) and 'ms_per_shot' in d and 'n' in d)
    s['decoder_cpu_h'] += cpu / 3600.0
    t['decoder_cpu_h'] += cpu / 3600.0
    s['summaries_read'] += 1

tot = {'jobs': sum(s['jobs'] for s in stages.values()),
       'queue_wall_h': sum(s['queue_wall_h'] for s in stages.values()),
       'decoder_cpu_h_decode_stages': sum(s['decoder_cpu_h'] for k, s in stages.items() if k != 'fit')}
for s in stages.values():
    for k in ('queue_wall_h', 'decoder_cpu_h'):
        s[k] = round(s[k], 3)
    for t in s['by_tag'].values():
        for k in ('queue_wall_h', 'decoder_cpu_h'):
            t[k] = round(t[k], 3)
out = {'made': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'what': "round 1's compute per stage (scout; read-only on decoder's files); see the docstring",
       'inputs': {'queue_state.json': hashlib.md5(raw).hexdigest(),
                  'decode summaries': {'n': sum(s['summaries_read'] for s in stages.values()),
                                       'md5_of_their_md5s_in_sorted_job_order': digest.hexdigest()}},
       'queue': {'started_last_instance': q.get('_started'), 'finished': q.get('_finished'),
                 'workers': q.get('_workers'), 'counts': q.get('_counts')},
       'planned_core_h': {'d3': 48.9, 'fit': 5.7, 'd5': 31.9, 'd7': 2.6, 'total': 89,
                          'source': 'code/decoder/README.md, Stages and cost'},
       'stages': stages, 'totals': {k: round(v, 3) if isinstance(v, float) else v for k, v in tot.items()},
       'missing_summaries': missing, 'seconds': round(time.time() - t0, 1)}
tmp = OUT + '.tmp'
with open(tmp, 'w') as f:
    json.dump(out, f, indent=1)
os.replace(tmp, OUT)
print(json.dumps({k: out[k] for k in ('totals', 'missing_summaries', 'seconds')}))
for k, s in stages.items():
    print(k, s['jobs'], s['attempts'], s['queue_wall_h'], s['decoder_cpu_h'], s['first_start'], s['last_end'])

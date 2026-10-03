cd /workspaces/is-there-a-sounds-like-ai-direction-511e-claude-opus-5-5/src
source ../.venv/bin/activate
while pgrep -f "steer.py main" >/dev/null; do sleep 10; done
python steer.py main >> ../logs/steer_main.log 2>&1
python steer.py base > ../logs/steer_base.log 2>&1
python score_gens.py gen_main.json > ../logs/score_main.log 2>&1
python score_gens.py gen_base.json > ../logs/score_base.log 2>&1
python readback.py scores_gen_main.parquet scores_gen_base.parquet > ../logs/readback.log 2>&1
python judge.py scores_gen_main.parquet local > ../logs/judge_local_main.log 2>&1
python judge.py scores_gen_base.parquet local > ../logs/judge_local_base.log 2>&1
echo CHAIN_DONE >> ../logs/chain1.done

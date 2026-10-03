cd /workspaces/is-there-a-sounds-like-ai-direction-511e-claude-opus-5-5/src
source ../.venv/bin/activate
while [ ! -f ../logs/chain2.done ]; do sleep 20; done
bash ../logs/chain3_cmds.sh

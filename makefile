# ---------- DEFINE CONSTANTS --------------------------------------------------------------------
DIR_CODE_ON_HOST = /home/philipp/Dokumente/Studium/Masterarbeit/BrainSignalDecodeProject
DIR_CODE_ON_CLUSTER = muelleph@kisbat3:/work/dlclarge1/muelleph-bdp/

EXCLUDE_STANDARD = --exclude='.git/*' --exclude='.git' \
    --exclude='*.lock' --exclude='.idea' --exclude='__pycache__' --exclude='*.pyc'
EXCLUDE_BDP = --exclude='*.out' --exclude='snapshots' --exclude='optlogs' --exclude='*.pbs'


# ---------- SYNCHRONIZATION COMMANDS ------------------------------------------------------------
sync_code2cluster:
	rsync -av ${EXCLUDE_STANDARD} ${EXCLUDE_BDP} --exclude='test*' \
		${DIR_CODE_ON_HOST} -e ssh ${DIR_CODE_ON_CLUSTER}

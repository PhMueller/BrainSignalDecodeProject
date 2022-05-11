# ---------- DEFINE CONSTANTS --------------------------------------------------------------------
BASE_HOST = ${HOME}/Dokumente/Code
DIR_CODE_ON_HOST = ${BASE_HOST}/BrainSignalDecodeProject
DIR_CODE_ON_CLUSTER = muelleph@kisbat3:/work/dlclarge1/muelleph-bdp

RES_DIR_ON_HOST = ${BASE_HOST}/BrainSignalResults
RES_DIR_ON_CLUSTER = muelleph@kisbat3:/work/dlclarge1/muelleph-bdp/results

EXCLUDE_STANDARD = --exclude='.git/*' --exclude='.git' --exclude='data' \
    --exclude='*.lock' --exclude='.idea' --exclude='__pycache__' --exclude='*.pyc'
EXCLUDE_BDP = --exclude='*.out' --exclude='snapshots' --exclude='optlogs' --exclude='*.pbs'


# ---------- SYNCHRONIZATION COMMANDS ------------------------------------------------------------
sync_code2cluster:
	rsync -av ${EXCLUDE_STANDARD} ${EXCLUDE_BDP} --exclude='test*' \
		${DIR_CODE_ON_HOST} -e ssh ${DIR_CODE_ON_CLUSTER}/

sync_res2home:
	rsync -av ${EXCLUDE_STANDARD} ${EXCLUDE_BDP} ${RES_DIR_ON_CLUSTER}/ -e ssh ${RES_DIR_ON_HOST}/

download_data:
	# You get the username and password at
	# https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tusz

	# Make sure to set the correct data path
	rsync -auxvL nedc@www.isip.piconepress.com:data/eeg/tuh_eeg_abnormal .
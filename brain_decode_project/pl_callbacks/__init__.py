from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

from brain_decode_project.pl_callbacks.callbacks import (
    PrintCallback,
    CheckpointEveryNSteps,
    CountTrainingTimeCallBack,
    SaveSnapshotCallback,
    StopWhenLimitIsReachedCallback,
    ValidateByTimeCallback,
)

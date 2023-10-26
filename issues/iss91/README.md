To run this issue 91, first generate a checkpoint with `train_checkpoint.py` or `train_ssa_checkpoint.py`.
This will create an experiment dir in `/data`.
During training, cancel training via ctl+c.
This will leave an unfinished experiment. 

Then edit `test_checkpoint_resume.py` to point to the appropriate experiment dir.
Then run the script.
This should resume the experiment. 
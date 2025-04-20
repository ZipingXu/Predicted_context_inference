To run simulation with environment failure1 for 5000 steps :
```
python run_exp.py -T 5000 --n_rep 100 --env failure1
```

You may visualize the results by running `failure1.ipynb`

rsync -av --exclude-from=sync_exclude.txt ~/Desktop/Research/Predicted_context_inference/ zipingxu@login.rc.fas.harvard.edu:/n/home03/zipingxu/Predicted_context_inference/
rsync -av zipingxu@login.rc.fas.harvard.edu:/n/home03/zipingxu/Predicted_context_inference/runs/ ~/Desktop/Research/Predicted_context_inference/runs/


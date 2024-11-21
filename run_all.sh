# run all

python run_exp.py --env failure1 --n_rep 1000 -T 10000 --p0 0.2 --lambda 0.1 --coverage_freq 100000
python run_exp.py --env failure2 --n_rep 1000 -T 10000 --p0 0.2 --lambda 0.1 --coverage_freq 100000
python run_exp.py --env random --n_rep 1000 -T 10000 --p0 0.2 --lambda 0.1 --coverage_freq 100000
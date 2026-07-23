@echo off
python run_reproducible_demo.py --out manuscript_results --epochs 10 --seeds 5 --moderate-test 2.2 --stress-test 6.0 --disturbance-stress 1.0 --display-seed 4 --animate --clean %*

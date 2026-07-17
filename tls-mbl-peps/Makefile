.PHONY: verify verify-jax test
verify:
	cd prototypes && python3 golden_3x3.py && python3 sdrg_3site.py && python3 bench_kernel.py timing 2 3 4
verify-jax:
	cd prototypes && python3 ad_phase2.py consistency && python3 ad_phase2.py adfd 4 && python3 ad_phase2.py adfd 2
test:
	pytest tests/ -x -q

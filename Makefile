.PHONY: verify verify-jax test
verify:
	cd prototypes && uv run python golden_3x3.py && uv run python sdrg_3site.py && uv run python bench_kernel.py timing 2 3 4
verify-jax:
	cd prototypes && uv run python ad_phase2.py consistency && uv run python ad_phase2.py adfd 4 && uv run python ad_phase2.py adfd 2
test:
	uv run pytest tests/ -x -q

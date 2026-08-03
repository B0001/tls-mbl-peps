# Kubernetes deployment

`tlsmbl run` as a batch Job. This is run-to-completion compute, not a service —
there is no Deployment, Service, or Ingress here by design.

```bash
docker build -t tls-mbl-peps:0.0.1 .
kubectl apply -k k8s/overlays/dev     # namespace tlsmbl-dev,  smoke config (L=4)
kubectl apply -k k8s/overlays/prod    # namespace tlsmbl-prod, pilot config (L=8)
```

| Path | Purpose |
|---|---|
| `base/job.yaml` | The solver Job, ConfigMap-mounted config, PVC at `/app/runs` |
| `base/run-config.yaml` | The **cluster** run config, mounted at `/etc/tlsmbl/run.yaml` |
| `base/pvc.yaml` | 20Gi RWO for the `.zarr` output |
| `overlays/dev` | Runs the image's baked `configs/smoke.yaml`, 2 CPU, 30 min deadline |
| `overlays/prod` | 4 CPU Guaranteed QoS, 24h deadline, 100Gi, `batch-high` priority |

## Why a separate `run-config.yaml`

It is deliberately **not** a copy of `configs/pilot_L8.yaml`. Kustomize's load
restrictor will not read files outside the kustomization directory, and a
duplicated config would silently drift from the original. Cluster runs also
differ on two axes:

- `run.workers` must track the pod's CPU **limit**, not a laptop's core count.
- `run.out` must resolve under `/app/runs`, where the PVC mounts.

The repo's own `configs/` are still baked into the image, so
`args: ["run", "configs/smoke.yaml"]` works without touching this file — that
is exactly what the dev overlay does.

## Threads and CPU move together

The image pins `OMP/OPENBLAS/MKL_NUM_THREADS=1`. Parallelism is process-level:
`run.workers` forks N realization workers, and without the pin each would also
spawn NCPU BLAS threads, oversubscribing the pod's limit by `workers × NCPU`.

**So `run.workers` and the Job's `cpu` must be the same number.** To scale a run
up, raise both. Raising `cpu` alone just leaves cores idle.

## Re-running

A Job's pod template is immutable, so re-running with a changed config needs the
old Job deleted first:

```bash
kubectl delete job tlsmbl-run-prod -n tlsmbl-prod && kubectl apply -k k8s/overlays/prod
```

The run is checkpointed (`env.checkpoint_rows`), so a retry after eviction
resumes rather than restarting — which is why prod sets `backoffLimit: 3`.

## Before deploying

- `priorityClassName: batch-high` in prod must already exist in the cluster, or
  the pod stays Pending on failed admission. Drop the line if you have no such
  class.
- Verify the `.zarr` afterwards with `tlsmbl verify` as a separate Job against
  the same claim — never concurrently with a run, since the PVC is RWO.

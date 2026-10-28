# Ablation YAML index (models/26) — proposed model = Ideal (HyperACE_Dyn)

## Keep (train / paper)

| ID | File | Content | Role |
|----|------|---------|------|
| A0 | `yolo26.yaml` | YOLO26 stock | Baseline |
| A1 | `yolo26-dyn.yaml` | + DynamicConv | +Dyn |
| A2 | `yolo26-ablate-carafe.yaml` | Dyn + CARAFE (PAN) | +CARAFE |
| A3 | `yolo26-ablate-hace.yaml` | Dyn + HyperACE_Dyn + FullPAD (nearest) | +HACE_Dyn (no CARAFE) |
| A4 | `yolo26-ablate-hace-dsc.yaml` | Dyn + HyperACE(DSC3k) + CARAFE + FullPAD | HACE DS vs Dyn |
| **Ideal** | **`yolo26-ideal.yaml`** | Dyn + **HyperACE_Dyn** + CARAFE + FullPAD | **Proposed** |

## Ablation logic

```
A0 YOLO26
 └─ A1 +Dyn
      ├─ A2 +CARAFE
      ├─ A3 +HyperACE_Dyn+FullPAD
      └─ Ideal = A3 path + CARAFE  (and A4 = Ideal but DSC3k inside HyperACE)
```

Claim a module only if Ideal (or the step that adds it) beats A0 on **PPE test** under the same protocol.

## Deleted (search losers / duplicates)

- `yolo26-ideal-hace-dynlo.yaml` → merged into `yolo26-ideal.yaml`
- `yolo26-ideal-carafe-hace*.yaml`, `yolo26-ideal-spdp3-carafe.yaml`
- `yolo26-dyn-spd*.yaml`, `yolo26-dyn-carafe.yaml` (→ `yolo26-ablate-carafe.yaml`)
- `yolo26-hyperace*.yaml`

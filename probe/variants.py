"""Scene variants of examples/average_spine.json for the D3D compile probe (throwaway)."""
import copy, json, re, sys
from pathlib import Path
src = json.loads(Path("examples/average_spine.json").read_text())
out = Path(sys.argv[1]); out.mkdir(exist_ok=True)
def save(name, nodes):
    d = copy.deepcopy(src); d["nodes"] = nodes; d["name"] = name
    (out / f"{name}.json").write_text(json.dumps(d))
N = src["nodes"]
save("v_full", N)
save("v_nodiscs", {k: v for k, v in N.items() if not k.startswith("disc_")})
rest = copy.deepcopy(N)
for k, v in rest.items():
    if v["block"] == "disc":
        for s in ("upper_v", "lower_v"):
            v["in"][s] = re.sub(r"_(align|pose)\.vertebra$", "_lm.vertebra", v["in"][s])
save("v_restlm", rest)
raw = copy.deepcopy(N)
for k, v in raw.items():
    if v["block"] == "disc":
        for s in ("upper", "lower"):
            v["in"][s] = re.sub(r"_(align|pose|smooth|warp)$", "", v["in"][s])
save("v_rawbones", raw)
save("v_annulus", {k: v for k, v in N.items() if not (k.startswith("disc_") and k.endswith("nucleus_out"))})
one = {k: v for k, v in N.items() if not k.startswith("disc_") or k.startswith("disc_L4_L5")}
save("v_onedisc", one)

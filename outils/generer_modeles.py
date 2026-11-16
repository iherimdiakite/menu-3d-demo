"""Génère les deux modèles 3D de SUBSTITUTION de la démo (pas de vraies captures de plats).
Usage : python demo/outils/generer_modeles.py   -> écrit demo/modeles/*.glb
Unités : mètres, taille réelle (nécessaire pour la réalité augmentée)."""
import json, struct, pathlib
import numpy as np

rng = np.random.default_rng(3)
OUT = pathlib.Path(__file__).resolve().parent.parent / "modeles"
_K, _PH = rng.normal(size=(6, 3)), rng.uniform(0, 6.28, 6)


def noise(P, freq):
    return np.mean(np.sin(P @ _K.T * freq + _PH), axis=-1)


def rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def grid(P, wrap):
    """P[nu, nv, 3] -> sommets + triangles ; wrap = boucle fermée sur u."""
    nu, nv = P.shape[:2]
    i = np.arange(nu if wrap else nu - 1)[:, None]
    j = np.arange(nv - 1)[None, :]
    i2 = (i + 1) % nu
    a, b, c, d = i * nv + j, i2 * nv + j, i2 * nv + j + 1, i * nv + j + 1
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return P.reshape(-1, 3).copy(), F


def lathe(profile, segs=48):
    r, y = np.asarray(profile, float).T
    th = np.linspace(0, 2 * np.pi, segs, endpoint=False)[:, None]
    return grid(np.stack([r * np.cos(th), y + 0 * th, r * np.sin(th)], -1), True)


def tube(pts, rad, nu=14):
    pts = np.asarray(pts, float)
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)[:, None]
    P = np.zeros((nu, len(pts), 3))
    for j, t in enumerate(tan):
        ref = np.array([0, 1.0, 0]) if abs(t[1]) < 0.9 else np.array([1.0, 0, 0])
        a = np.cross(t, ref); a /= np.linalg.norm(a)
        P[:, j] = pts[j] + rad[j] * (np.cos(th) * a + np.sin(th) * np.cross(t, a))
    return grid(P, True)


_BOX_V = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) - 0.5
_BOX_Q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
_BOX_F = np.array([t for a, b, c, d in _BOX_Q for t in ((a, b, c), (a, c, d))])


def box(size, R, pos):
    """Cube à facettes (sommets non partagés)."""
    V = (_BOX_V * size) @ R.T + pos
    return V[_BOX_F].reshape(-1, 3), np.arange(36).reshape(-1, 3)


def rand_rot():
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def merge(meshes):
    """[(V, F, C)] -> (V, F, C)"""
    Vs, Fs, Cs, n = [], [], [], 0
    for V, F, C in meshes:
        Vs.append(V); Fs.append(F + n); Cs.append(np.broadcast_to(C, V.shape)); n += len(V)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Cs)


def normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]


def part(name, V, F, base, C=1.0, rough=0.6, alpha=1.0, orient=True):
    """base : couleur sRGB ; C : multiplicateur par sommet (<= 1).
    La couleur de base reste lisible seule (le Quick Look iOS peut ignorer les couleurs de sommet)."""
    N = normals(V, F)
    if orient and np.mean(np.sum(N * (V - V.mean(0)), axis=1)) < 0:
        F, N = F[:, ::-1], -N
    return dict(name=name, V=V.astype(np.float32), F=F, N=N.astype(np.float32),
                C=np.broadcast_to(np.clip(C, 0, 1), V.shape), base=base, rough=rough, alpha=alpha)


def write_glb(path, parts):
    blob = bytearray()
    g = dict(asset={"version": "2.0", "generator": "Weilth demo - modele de substitution"},
             scene=0, scenes=[{"nodes": list(range(len(parts)))}],
             nodes=[], meshes=[], materials=[], accessors=[], bufferViews=[], buffers=[])

    def acc(arr, ctype, typ, target, norm=False, minmax=False):
        while len(blob) % 4:
            blob.append(0)
        g["bufferViews"].append(dict(buffer=0, byteOffset=len(blob), byteLength=arr.nbytes, target=target))
        blob.extend(arr.tobytes())
        a = dict(bufferView=len(g["bufferViews"]) - 1, componentType=ctype, count=len(arr), type=typ)
        if norm:
            a["normalized"] = True
        if minmax:
            a["min"], a["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        g["accessors"].append(a)
        return len(g["accessors"]) - 1

    for i, p in enumerate(parts):
        assert np.isfinite(p["V"]).all() and np.isfinite(p["N"]).all(), p["name"]
        assert p["F"].min() >= 0 and p["F"].max() < len(p["V"]), p["name"]
        rgba = np.concatenate([p["C"] * 255, np.full((len(p["V"]), 1), 255)], 1).round().astype(np.uint8)
        small = len(p["V"]) < 65536
        idx = p["F"].reshape(-1).astype(np.uint16 if small else np.uint32)
        prim = dict(material=i, indices=acc(idx, 5123 if small else 5125, "SCALAR", 34963),
                    attributes=dict(POSITION=acc(p["V"], 5126, "VEC3", 34962, minmax=True),
                                    NORMAL=acc(p["N"], 5126, "VEC3", 34962),
                                    COLOR_0=acc(rgba, 5121, "VEC4", 34962, norm=True)))
        lin = [round(float(c) ** 2.2, 4) for c in p["base"]] + [p["alpha"]]
        m = dict(name=p["name"], doubleSided=True,
                 pbrMetallicRoughness=dict(baseColorFactor=lin, metallicFactor=0, roughnessFactor=p["rough"]))
        if p["alpha"] < 1:
            m["alphaMode"] = "BLEND"
        g["materials"].append(m)
        g["meshes"].append(dict(name=p["name"], primitives=[prim]))
        g["nodes"].append(dict(name=p["name"], mesh=i))
    while len(blob) % 4:
        blob.append(0)
    g["buffers"].append(dict(byteLength=len(blob)))
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    data = (struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(blob))
            + struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    assert len(data) < 3_000_000, "modèle trop lourd pour la data mobile"
    print(f"\n{path.name}: {len(data) / 1e6:.2f} Mo, {sum(len(p['V']) for p in parts)} sommets")
    for p in parts:  # repères pour placer les étiquettes dans index.html
        print(f"  {p['name']:<16} centre {np.round(p['V'].mean(0), 3)}  haut {p['V'][:, 1].max():.3f}")


# ---------------------------------------------------------------- attiéké poisson braisé
PLATE_Y = 0.006  # hauteur du fond de l'assiette


def dish():
    parts = []
    V, F = lathe([(0, PLATE_Y), (0.10, PLATE_Y), (0.118, 0.011), (0.15, 0.024), (0.152, 0.022),
                  (0.115, 0.004), (0.07, 0), (0, 0)], 72)
    parts.append(part("assiette", V, F, (0.96, 0.95, 0.92), rough=0.22))

    # attiéké : dôme irrégulier, grain donné par un bruit par sommet
    th = np.linspace(0, 2 * np.pi, 150, endpoint=False)[:, None]
    v = np.linspace(0, 1, 50)[None, :]
    r = 0.048 * (1 + 0.10 * np.sin(3 * th + 1) + 0.06 * np.sin(5 * th)) * v
    P = np.stack([r * np.cos(th), 0.036 * (1 - v ** 2.2) + 0 * th, r * np.sin(th)], -1)
    P += rng.normal(0, 0.0011, P.shape) * (v[..., None] > 0)
    V, F = grid(P, True)
    C = rng.uniform(0.62, 1.0, (len(V), 1)) * np.array([1, 0.97, 0.86])
    parts.append(part("attieke", V + [-0.04, PLATE_Y, -0.05], F, (0.98, 0.89, 0.6), C, rough=0.95))

    # poisson braisé : modelé debout puis couché sur le flanc
    L, ns, nu = 0.19, 140, 40
    s = np.linspace(0, 1, ns)
    prof = np.interp(s, [0, .04, .12, .3, .55, .78, .9, 1], [.02, .35, .7, 1, .92, .45, .2, .16])
    hh, ww = 0.038 * prof, 0.015 * np.minimum(prof * 1.1, 1)
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)[:, None]
    x = (s - 0.5) * L + 0 * th
    body = grid(np.stack([x, hh * np.sin(th), ww * np.cos(th)], -1), True)
    Vb = body[0]
    char = np.clip(noise(Vb, 90) * 1.6 + 0.25, 0, 1)                      # zones grillées
    slash = (np.abs((Vb[:, 0] + 0.6 * Vb[:, 1]) % 0.03 - 0.015) < 0.0025) & (np.abs(Vb[:, 0]) < 0.05)
    Cb = (1 - 0.6 * char)[:, None] * np.array([1, 0.95, 0.85]) * np.where(slash, 0.3, 1)[:, None]
    a = np.linspace(0, 1, 12)[:, None]; b = np.linspace(-1, 1, 15)[None, :]
    tail = grid(np.stack([L / 2 - 0.012 + 0.05 * a - 0.02 * a * (1 - np.abs(b)),
                          b * (0.006 + 0.036 * a), 0 * a * b], -1), False)
    t = np.linspace(0, 1, 40)[:, None]; e = np.linspace(0, 1, 4)[None, :]
    sd = 0.22 + 0.5 * t
    dorsal = grid(np.stack([(sd - 0.5) * L + 0 * e, np.interp(sd, s, hh) - 0.002 + e * 0.014 * np.sin(np.pi * t) ** 0.5,
                            0 * t * e], -1), False)
    V, F, C = merge([(*body, Cb), (*tail, 0.42), (*dorsal, 0.42)])
    M = ry(np.radians(12)) @ rx(np.radians(90))        # flanc vers le haut, dos vers le centre
    pos = np.array([0.005, PLATE_Y + 0.014, 0.055])
    parts.append(part("poisson", V @ M.T + pos, F, (0.86, 0.56, 0.26), C, rough=0.5, orient=False))
    eye = lathe([(0.0055 * np.sin(q), -0.0055 * np.cos(q)) for q in np.linspace(0, np.pi, 7)], 12)
    parts.append(part("poisson_oeil", (eye[0] + [(0.1 - 0.5) * L, 0.008, -0.0105]) @ M.T + pos, eye[1],
                      (0.93, 0.92, 0.88), rough=0.3))

    # sauce tomate-oignon : petits dés
    cubes = {"tomate": [], "oignon": [], "herbes": []}
    for _ in range(90):
        kind = rng.choice(["tomate", "oignon", "herbes"], p=[.55, .35, .10])
        size = rng.uniform(0.005, 0.009) * (0.6 if kind == "herbes" else 1)
        rr, ang = 0.028 * np.sqrt(rng.uniform()), rng.uniform(0, 6.28)
        y = PLATE_Y + size / 2 + 0.018 * (1 - (rr / 0.028) ** 2) * rng.uniform()
        Vc, Fc = box(size, rand_rot(), [0.045 + rr * np.cos(ang), y, -0.06 + rr * np.sin(ang)])
        cubes[kind].append((Vc, Fc, np.full(3, rng.uniform(0.7, 1.0))))
    for name, base, rough in [("tomate", (0.86, 0.17, 0.1), 0.35), ("oignon", (0.96, 0.9, 0.93), 0.4),
                              ("herbes", (0.3, 0.56, 0.16), 0.7)]:
        V, F, C = merge(cubes[name])
        parts.append(part("sauce_" + name, V, F, base, C, rough=rough, orient=False))

    # alloco : rondelles de banane plantain frite
    sl = []
    for k in range(6):
        V, F = lathe([(0, 0), (0.9, 0), (1, 0.25), (1, 0.75), (0.9, 1), (0, 1)], 28)
        edge = np.hypot(V[:, 0], V[:, 2])
        C = (1 - 0.45 * edge ** 3 * rng.uniform(0.5, 1))[:, None] * np.array([1, 0.9, 0.7]) * np.ones((len(V), 3))
        V = V * [0.02, 0.007, 0.014]
        R = ry(rng.uniform(0, 6.28)) @ rx(rng.uniform(-0.25, 0.25))
        sl.append((V @ R.T + [0.082 + rng.uniform(-.014, .014), PLATE_Y + 0.003 + 0.0035 * k,
                              -0.012 + rng.uniform(-.014, .014)], F, C))
    V, F, C = merge(sl)
    parts.append(part("alloco", V, F, (0.93, 0.64, 0.17), C, rough=0.45, orient=False))

    # piment
    t = np.linspace(0, 1, 30)
    path = np.stack([0.06 * (t - 0.5), 0 * t, 0.012 * (2 * t - 1) ** 2], -1)
    rad = 0.0065 * (1 - t) ** 0.6 * np.minimum(1, t * 8 + 0.3) + 0.0003
    M, pos = ry(np.radians(20)), np.array([0.0, PLATE_Y + 0.0065, -0.012])
    V, F = tube(path, rad)
    parts.append(part("piment", V @ M.T + pos, F, (0.82, 0.08, 0.05), rough=0.2))
    V, F = tube(path[0] + np.outer(np.linspace(0, 1, 4), [-0.012, 0.002, 0]), np.full(4, 0.0014), 8)
    parts.append(part("piment_tige", V @ M.T + pos, F, (0.3, 0.5, 0.15), rough=0.6))
    write_glb(OUT / "attieke-poisson.glb", parts)


# ---------------------------------------------------------------- cocktail au bissap
def cocktail():
    H, RB, RT, WALL, BASE, FILL = 0.15, 0.030, 0.036, 0.002, 0.012, 0.125
    r_in = lambda y: RB + (RT - RB) * y / H - WALL
    parts = []

    # liquide (opaque, dans le verre)
    ys = np.linspace(BASE + 0.0005, FILL, 12)
    V, F = lathe([(0, ys[0])] + [(r_in(y) - 0.0004, y) for y in ys] + [(0, FILL)], 48)
    C = (0.55 + 0.45 * (V[:, 1] - BASE) / (FILL - BASE))[:, None] * np.ones(3)
    parts.append(part("bissap", V, F, (0.5, 0.02, 0.1), C, rough=0.08))

    # paille
    p0, p1 = np.array([0.008, 0.016, -0.012]), np.array([-0.02, 0.2, -0.03])
    V, F = tube(p0 + np.outer(np.linspace(0, 1, 6), p1 - p0), np.full(6, 0.003), 14)
    parts.append(part("paille", V, F, (0.86, 0.72, 0.48), rough=0.8))

    # rondelle de citron vert, à cheval sur le bord
    R, T = 0.02, 0.0022
    rs = np.linspace(0, R, 10)
    V, F = lathe([(q, -T) for q in rs] + [(R, -T * 0.4), (R, T * 0.4)] + [(q, T) for q in rs[::-1]], 64)
    rho, ang = np.hypot(V[:, 0], V[:, 2]) / R, np.arctan2(V[:, 2], V[:, 0])
    flesh, pith, rind = np.array([0.72, 0.88, 0.38]), np.array([1, 1, 0.92]), np.array([0.25, 0.55, 0.1])
    C = np.where(((np.abs(np.sin(4 * ang)) < 0.13) | (rho < 0.1))[:, None], pith, flesh)
    C = np.where((rho > 0.8)[:, None], pith, C)
    C = np.where((rho > 0.9)[:, None], rind, C)
    parts.append(part("citron_vert", V @ rx(np.radians(90)).T + [RT - 0.003, H + 0.004, 0], F,
                      (0.97, 0.98, 0.85), C, rough=0.4))

    # menthe
    u = np.linspace(0, 1, 14)[:, None]; v = np.linspace(-1, 1, 7)[None, :]
    leaves = []
    for k in range(5):
        ln = rng.uniform(0.028, 0.038)
        P = np.stack([0.3 * ln * np.sin(np.pi * u) ** 0.6 * v, 0.25 * ln * u ** 2 + 0.003 * np.abs(v), ln * u + 0 * v], -1)
        V, F = grid(P, False)
        C = np.where(np.abs(V[:, 0]) < 0.0012, 1.0, 0.72)[:, None] * np.ones(3)
        R = ry(k * 2 * np.pi / 5 + 0.4) @ rx(np.radians(-rng.uniform(40, 65)))
        leaves.append((V @ R.T + [-0.012, 0.127, 0.006], F, C))
    V, F, C = merge(leaves)
    parts.append(part("menthe", V, F, (0.3, 0.62, 0.2), C, rough=0.6, orient=False))

    # glaçons, à fleur de liquide
    ice = [(*box(0.02, rand_rot(), [0.013 * np.cos(q) + 0.002, 0.121 + 0.004 * (i % 2), 0.013 * np.sin(q) + 0.004]), 1.0)
           for i, q in enumerate([0.4, 2.3, 4.4])]
    V, F, C = merge(ice)
    parts.append(part("glacons", V, F, (0.93, 0.97, 1.0), C, rough=0.05, alpha=0.6, orient=False))

    # verre (dessiné en dernier : transparent)
    V, F = lathe([(0, 0), (RB, 0), (RT, H), (RT - WALL, H), (r_in(BASE), BASE), (0, BASE)], 64)
    parts.append(part("verre", V, F, (0.9, 0.96, 1.0), rough=0.04, alpha=0.25))
    write_glb(OUT / "cocktail-bissap.glb", parts)


if __name__ == "__main__":
    dish()
    cocktail()

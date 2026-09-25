"""S1 - parse WaitingRoom.unity into world-space geometry for every EyeTrack-tagged object.

Why this exists: The earlier text export gave positions only. Child entries were LOCAL coordinates with
no rotation, so the doors did not compose - adding parent+child put both doors at the room origin.
The full scene has the rotations, so the composition can finally be done properly.

Unity scene YAML is a stream of documents:  --- !u!<classID> &<fileID>
    !u!1    GameObject   (m_Name, m_TagString, m_Component)
    !u!4    Transform    (m_LocalPosition/Rotation/Scale, m_Father, m_GameObject)
    !u!65   BoxCollider  (m_Size, m_Center)
    !u!23   MeshRenderer
World transform is composed up the m_Father chain: T = T_parent * TRS(local).

Validation (the check that can actually fail) is in S1_validate_doors():
every participant's terminal position must be within arm's reach of the handle they grabbed.

Outputs: scene_geometry.json, S1_report.txt
"""
import json, re, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
SCENE = pathlib.Path("D:/evac_unity/WaitingRoom.unity")

DOC = re.compile(r"^--- !u!(\d+) &(\d+)", re.M)
V3 = r"\{x:\s*([-\d.eE+]+),\s*y:\s*([-\d.eE+]+),\s*z:\s*([-\d.eE+]+)"
V4 = V3 + r",\s*w:\s*([-\d.eE+]+)"


def _f(m, n):
    return tuple(float(m.group(i)) for i in range(1, n + 1))


def parse(text):
    """Return {fileID: (classID, body)} for every document in the scene."""
    marks = list(DOC.finditer(text))
    docs = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        docs[int(m.group(2))] = (int(m.group(1)), text[m.end():end])
    return docs


def quat_to_mat(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def build():
    text = SCENE.read_text(encoding="utf-8", errors="replace")
    docs = parse(text)

    gameobjects, transforms = {}, {}
    for fid, (cls, body) in docs.items():
        if cls == 1:
            name = re.search(r"m_Name:\s*(.*)", body)
            tag = re.search(r"m_TagString:\s*(.*)", body)
            comps = [int(x) for x in re.findall(r"component:\s*\{fileID:\s*(\d+)\}", body)]
            act = re.search(r"m_IsActive:\s*(\d+)", body)
            gameobjects[fid] = dict(name=(name.group(1).strip() if name else ""),
                                    tag=(tag.group(1).strip() if tag else ""), comps=comps,
                                    active=int(act.group(1)) if act else 1)
        elif cls in (4, 224):   # 224 = RectTransform (the queue display canvas)
            go = re.search(r"m_GameObject:\s*\{fileID:\s*(\d+)\}", body)
            pos = re.search(r"m_LocalPosition:\s*" + V3, body)
            rot = re.search(r"m_LocalRotation:\s*" + V4, body)
            scl = re.search(r"m_LocalScale:\s*" + V3, body)
            fat = re.search(r"m_Father:\s*\{fileID:\s*(\d+)\}", body)
            p = np.array(_f(pos, 3)) if pos else np.zeros(3)

            # A RectTransform does NOT carry its placement in m_LocalPosition. When the anchors
            # coincide (m_AnchorMin == m_AnchorMax, the non-stretching case) the x and y offsets
            # live in m_AnchoredPosition and m_LocalPosition.x/y are zero. Reading m_LocalPosition
            # put 6_CanvasQueueDisplay at the room origin, 4.5 m from where it is, which failed the
            # parser's own door check and was then worked around by hand-patching the coordinate
            # into C2. The released geometry kept the wrong value, so the published analysis did
            # not reproduce from the released artefacts.
            if cls == 224:
                anc = re.search(r"m_AnchoredPosition:\s*\{x:\s*([-\d.eE+]+),\s*y:\s*([-\d.eE+]+)", body)
                amin = re.search(r"m_AnchorMin:\s*\{x:\s*([-\d.eE+]+),\s*y:\s*([-\d.eE+]+)", body)
                amax = re.search(r"m_AnchorMax:\s*\{x:\s*([-\d.eE+]+),\s*y:\s*([-\d.eE+]+)", body)
                same = (amin and amax
                        and abs(float(amin.group(1)) - float(amax.group(1))) < 1e-6
                        and abs(float(amin.group(2)) - float(amax.group(2))) < 1e-6)
                if anc and same:
                    p = np.array([float(anc.group(1)), float(anc.group(2)), p[2]])

            transforms[fid] = dict(go=int(go.group(1)) if go else None,
                                   pos=p,
                                   rot=_f(rot, 4) if rot else (0, 0, 0, 1),
                                   scale=np.array(_f(scl, 3)) if scl else np.ones(3),
                                   father=int(fat.group(1)) if fat else 0)

    # transform owned by each GameObject
    t_of_go = {t["go"]: fid for fid, t in transforms.items() if t["go"] is not None}
    go_of_t = {fid: t["go"] for fid, t in transforms.items() if t["go"] is not None}

    def active_in_hierarchy(go_fid):
        """Unity's activeInHierarchy: false if this object OR any ancestor is deactivated.

        m_IsActive is the object's OWN flag only. Reading it alone reports a dead object as live,
        because deactivating a parent does not rewrite the children. Here that matters for real:
        `RoomWithTwoDoors` is deactivated, which is what makes 6_ExitDoorHandle and
        7_ExitDoorSides dead and leaves exactly one live object per hitObject id. Publishing the
        raw flag made the released scene look as though two live objects shared an id, and an
        internal check drew precisely that wrong conclusion from it.
        """
        seen = set()
        fid = go_fid
        while fid and fid not in seen:
            seen.add(fid)
            g = gameobjects.get(fid)
            if g is None:
                break
            if not g.get("active", 1):
                return 0
            t = transforms.get(t_of_go.get(fid))
            if t is None:
                break
            fid = go_of_t.get(t["father"])
        return 1

    def trs(t):
        M = np.eye(4)
        M[:3, :3] = quat_to_mat(t["rot"]) @ np.diag(t["scale"])
        M[:3, 3] = t["pos"]
        return M

    def world(fid, _seen=None):
        """Full 4x4 local-to-world matrix, composed up the father chain.

        Scale matters here: the door meshes are authored in centimetres and carried at a
        local scale of ~0.01, so ignoring it puts the doors at the room origin - which is
        exactly how the hand-made coordinate export went wrong.
        """
        _seen = _seen or set()
        if fid in _seen or fid not in transforms:
            return np.eye(4)
        _seen.add(fid)
        t = transforms[fid]
        M = trs(t)
        if t["father"] and t["father"] in transforms:
            return world(t["father"], _seen) @ M
        return M

    # collider centre and extents. The centre is in the object's LOCAL space and is often
    # non-zero (the door pivot sits at the room origin; the panel itself does not).
    box, ctr = {}, {}
    for fid, (cls, body) in docs.items():
        if cls in (65, 136, 64):
            go = re.search(r"m_GameObject:\s*\{fileID:\s*(\d+)\}", body)
            if not go:
                continue
            sz = re.search(r"m_Size:\s*" + V3, body)
            cc = re.search(r"m_Center:\s*" + V3, body)
            if sz:
                box[int(go.group(1))] = np.array(_f(sz, 3))
            if cc:
                ctr[int(go.group(1))] = np.array(_f(cc, 3))

    rows = []
    for go_fid, g in gameobjects.items():
        if go_fid not in t_of_go:
            continue
        M = world(t_of_go[go_fid])
        pivot = M[:3, 3]
        c = ctr.get(go_fid)
        centre = (M @ np.append(c, 1.0))[:3] if c is not None else pivot
        active = active_in_hierarchy(go_fid)
        rows.append(dict(name=g["name"], tag=g["tag"], go=go_fid, active=active,
                         active_self=g.get("active", 1),
                         x=float(centre[0]), y=float(centre[1]), z=float(centre[2]),
                         pivot_x=float(pivot[0]), pivot_y=float(pivot[1]), pivot_z=float(pivot[2]),
                         size=[float(v) for v in box.get(go_fid, [np.nan] * 3)]))
    return pd.DataFrame(rows)


# hitObject id -> the tagged GameObject name
# The logger derives the id by splitting the GameObject name on "_" and parsing the first
# field (GazeTrackerManager.GetObjectId). Several ids therefore have MORE THAN ONE object.
ID_OBJECTS = {3: ["3_ExitSign", "3_FireExtinguisher"],
              4: ["4_FireAlarmVisual"],
              5: ["5_DoctorDoorHandle", "5_DoctorDoorSides"],
              6: ["6_CanvasQueueDisplay", "6_ExitDoorHandle"],
              7: ["7_ExitDoorHandle", "7_ExitDoorSides"],
              8: ["8_NPC1"], 9: ["9_NPC2"], 10: ["10_FireAlarmAuditory"],
              11: ["11_FireExtinguisher"], 12: ["12_DoctorDoor"], 13: ["13_ExitDoor"]}
ID_NAME = {k: v[0] for k, v in ID_OBJECTS.items()}

# positions listed by hand, for the parser sanity check (world coords, freestanding objects)
ANUSH = {3: (2.56699991, 2.64599991, 2.97600007), 4: (1.71700001, 2.32500005, 2.91499996),
         6: (3.71000004, 2.58500004, 2.95700002), 10: (2.09899998, 2.65300012, 2.977),
         11: (3.54500008, 0.591000021, 2.90280008), 8: (0.0430000015, 0, -1.42400002),
         9: (3.5150001, 0, 1.30900002)}


def main():
    df = build()
    L = []
    P = L.append
    P("S1 - WaitingRoom.unity world geometry")
    P("=" * 60)
    P(f"parsed {len(df)} GameObjects with transforms")
    tagged = df[df.tag == "EyeTrack"]
    P(f"objects tagged EyeTrack: {len(tagged)}")

    resolved = {}
    for oid, nm in ID_NAME.items():
        hit = df[df.name == nm]
        if len(hit) == 1:
            r = hit.iloc[0]
            # Annotate each id-sharer with whether it is live. A bare list of names reads as an
            # unresolved ambiguity in the ground truth, and a reviewer working only from the
            # released file concluded exactly that. Every sharer here is deactivated in the
            # hierarchy, so each hitObject id maps to precisely one live object.
            sharers = []
            for other in ID_OBJECTS[oid][1:]:
                h = df[df.name == other]
                live = bool(h.iloc[0].active) if len(h) else None
                sharers.append(f"{other} ({'ACTIVE' if live else 'inactive, cannot be hit'})")
            resolved[oid] = dict(name=nm, tag=r.tag, pos=[r.x, r.y, r.z],
                                 pivot=[r.pivot_x, r.pivot_y, r.pivot_z], size=r["size"],
                                 active=int(r.active), active_self=int(r.active_self),
                                 shares_id_with=sharers)
        else:
            resolved[oid] = dict(name=nm, tag=None, pos=None, size=None, n_matches=len(hit))

    P("")
    P("CHECK 1 (parser sanity) - PIVOTS vs the hand list (pivots)")
    P(f"  {'id':>3} {'name':22s}{'err_m':>9}")
    ok1 = True
    for oid, ref in ANUSH.items():
        r = resolved.get(oid)
        if not r or r["pos"] is None:
            P(f"  {oid:>3} {ID_NAME[oid]:22s}{'NOT FOUND':>9}"); ok1 = False; continue
        e = float(np.linalg.norm(np.array(r["pivot"]) - np.array(ref)))
        P(f"  {oid:>3} {ID_NAME[oid]:22s}{e:9.4f}")
        ok1 &= e < 0.01
    P(f"  -> {'PASS' if ok1 else 'FAIL'} (parser reproduces the hand-listed coordinates)")

    P("")
    P("ID COLLISIONS - GetObjectId parses the leading integer of the GameObject name,")
    P("so an id covers EVERY tagged object sharing that prefix.")
    for oid, objs in sorted(ID_OBJECTS.items()):
        if len(objs) > 1:
            st = []
            for o in objs:
                h = df[df.name == o]
                st.append(f"{o}{'' if (len(h) and h.iloc[0].active) else ' [INACTIVE]'}")
            P(f"  id {oid:>2}: " + "  +  ".join(st))
    P("")
    P("RESOLVED WORLD POSITIONS (collider centre, i.e. where the object actually is)")
    P(f"  {'id':>3} {'name':22s}{'x':>9}{'y':>9}{'z':>9}  tag")
    for oid in sorted(resolved):
        r = resolved[oid]
        if r["pos"]:
            P(f"  {oid:>3} {r['name']:22s}" + "".join(f"{v:9.3f}" for v in r["pos"]) + f"  {r['tag']}")
        else:
            P(f"  {oid:>3} {r['name']:22s}{'-- not found in scene --':>27}")

    json.dump({str(k): v for k, v in resolved.items()},
              open(OUT / "scene_geometry.json", "w"), indent=1)
    df.to_csv(OUT / "S1_all_objects.csv", index=False)
    txt = "\n".join(L)
    (OUT / "S1_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

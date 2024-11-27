def parse_obj(file):
    vertexList = []     # 정점 좌표 리스트
    textureList = []    # 텍스처 좌표 리스트
    faceList = []       # 면 정보 리스트

    with open(file, "r") as objFile:
        for line in objFile:
            if not line.strip():  # 빈 줄 무시
                continue
            split = line.split()
            if split[0] == "#":  # 주석 무시
                continue
            elif split[0] == "v":
                x, y, z = map(float, split[1:])
                vertexList.append([x, y, z])
            elif split[0] == "vt":
                u, v = map(float, split[1:3])
                textureList.append([u, v])
            elif split[0] == "f":
                face = []
                for vertex in split[1:]:
                    v_data = vertex.split("/")
                    v_idx = int(v_data[0]) if v_data[0] else None
                    vt_idx = int(v_data[1]) if len(v_data) > 1 and v_data[1] else None
                    vn_idx = int(v_data[2]) if len(v_data) > 2 and v_data[2] else None
                    face.append([v_idx, vt_idx, vn_idx])
                faceList.append(face)

    print(f"{len(faceList)} faces")
    print(f"{len(vertexList)} vertices")
    print(f"{len(textureList)} texture vertices")

    return vertexList, textureList, faceList
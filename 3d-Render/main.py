import argparse
from PIL import Image, ImageOps
import render as r
import objload as objload
import os

def main():
    folder_name = "models"
    model_name = "Mustang" # "LibertStatue" ### CHANGE TO YOUR 3D!
    obj_file = f"{folder_name}/{model_name}.obj"
    tga_file = f"{folder_name}/{model_name}.tga"
    
    width, height = 800, 800
    print(f"Rendering with Image Size: Width = {width}, Height = {height}")
    
    if not os.path.exists(obj_file):
        print(f"There's no OBJ file '{obj_file}'")
        return

    # 이미지 생성
    img = Image.new("RGB", (width, height), "black")
    pixels = img.load()

    vertices, texture_vertices, faces = objload.parse_obj(obj_file)

    # TGA 파일 (OBG만 있고 TGA 없어도 됩니다!)
    if os.path.exists(tga_file):
        print(f"With TGA...")
        texture = Image.open(tga_file)
        # TGA 파일을 준비했는데, 결과 비트맵이 이상하다면 이 rotate를 지우거나 loadobj파일을 약간 수정해보세요.. 
        # 그럼에도 수정되지 않는다면, obj file의 vertices의 min, max value가 0, 1인지 확인하고, 아니라면 obj만 사용해서 돌려봅시다! 
        texture = texture.rotate(180)
        texture_dim = texture.size
        texture_array = texture.load()
        r.render_shaded(pixels, vertices, texture_vertices, faces, texture_array, texture_dim, width, height)
        output_file = f"{folder_name}/{model_name}_rendered_TGAo.bmp"
    else:
        print(f"Without TGA...")
    
        # 예외 케이스: vertices가 -1~1 범위를 벗어나는 경우
        vertex_min = [min(v[i] for v in vertices) for i in range(3)]
        vertex_max = [max(v[i] for v in vertices) for i in range(3)]
    
        # 모델 중심 계산
        center = [(vertex_min[i] + vertex_max[i]) / 2 for i in range(3)]
    
        # 모델 크기(최대 반지름) 계산
        max_extent = max(vertex_max[i] - vertex_min[i] for i in range(3))
    
        # 정규화: 중심을 기준으로 [-1, 1] 범위로 변환
        if max_extent > 0:
            print("Normalizing vertices to [-1, 1] range while preserving shape...")
            vertices = [
                [
                    2 * (v[i] - center[i]) / max_extent
                    for i in range(3)
                ]
                for v in vertices
            ]
        
            r.render_shaded(pixels, vertices, None, faces, None, None, width, height)
            output_file = f"{folder_name}/{model_name}_rendered_TGAx.bmp"


    # 렌더링 결과 저장
    img = ImageOps.flip(img) 
    img.save(output_file)
    print(f"Saved rendered image to {output_file}")


if __name__ == "__main__":
    main()

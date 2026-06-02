import argparse
import os
from pathlib import Path


CAMERAS = ("main", "A", "B", "C", "D")


def normalize_camera(value):
    camera = value.strip()
    camera = "main" if camera.lower() == "main" else camera.upper()

    if camera not in CAMERAS:
        raise ValueError(f"未知摄像头：{camera}，可选值：{', '.join(CAMERAS)}")

    return camera


def resolve_source_path(value, source_name):
    source_path = Path(value.strip().strip('"')).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"{source_name}不存在：{source_path}")

    return source_path


def parse_camera_source(value, source_name):
    if "=" not in value:
        raise ValueError(f"多路{source_name}参数格式应为 camera=路径，例如 A=demo.mp4")

    camera, source = value.split("=", 1)
    return normalize_camera(camera), resolve_source_path(source, source_name)


def collect_named_sources(args, suffix):
    return {
        "main": getattr(args, f"main_{suffix}"),
        "A": getattr(args, f"a_{suffix}"),
        "B": getattr(args, f"b_{suffix}"),
        "C": getattr(args, f"c_{suffix}"),
        "D": getattr(args, f"d_{suffix}"),
    }


def set_source_env(camera, source_type, source_path):
    os.environ[f"STATIC_CAMERA_{camera.upper()}_{source_type.upper()}"] = str(source_path)


def main():
    parser = argparse.ArgumentParser(
        description="用图片或视频文件模拟项目中的摄像头画面。视频会循环播放，更适合测试目标跟踪和累计车流统计。"
    )
    parser.add_argument(
        "--camera",
        default="main",
        choices=CAMERAS,
        help="单路模式下要替换的摄像头。"
    )
    parser.add_argument("--image", help="单路模式下用于模拟摄像头画面的图片路径。")
    parser.add_argument("--video", help="单路模式下用于模拟摄像头画面的视频路径，循环播放。")
    parser.add_argument(
        "--camera-image",
        action="append",
        default=[],
        metavar="CAMERA=IMAGE",
        help="多路图片模式，可重复传入，例如 --camera-image main=a.jpg --camera-image A=b.jpg。"
    )
    parser.add_argument(
        "--camera-video",
        action="append",
        default=[],
        metavar="CAMERA=VIDEO",
        help="多路视频模式，可重复传入，例如 --camera-video main=a.mp4 --camera-video A=b.mp4。"
    )
    parser.add_argument("--main-image", help="首页任意地点监测画面图片。")
    parser.add_argument("--a-image", help="至臻楼摄像头画面图片。")
    parser.add_argument("--b-image", help="友园15号楼摄像头画面图片。")
    parser.add_argument("--c-image", help="食堂摄像头画面图片。")
    parser.add_argument("--d-image", help="安楼摄像头画面图片。")
    parser.add_argument("--main-video", help="首页任意地点监测画面视频。")
    parser.add_argument("--a-video", help="至臻楼摄像头画面视频。")
    parser.add_argument("--b-video", help="友园15号楼摄像头画面视频。")
    parser.add_argument("--c-video", help="食堂摄像头画面视频。")
    parser.add_argument("--d-video", help="安楼摄像头画面视频。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)

    args = parser.parse_args()
    camera_images = {}
    camera_videos = {}

    if args.image:
        camera_images[normalize_camera(args.camera)] = resolve_source_path(args.image, "图片")

    if args.video:
        camera_videos[normalize_camera(args.camera)] = resolve_source_path(args.video, "视频")

    for value in args.camera_image:
        camera, image_path = parse_camera_source(value, "图片")
        camera_images[camera] = image_path

    for value in args.camera_video:
        camera, video_path = parse_camera_source(value, "视频")
        camera_videos[camera] = video_path

    for camera, image in collect_named_sources(args, "image").items():
        if image:
            camera_images[camera] = resolve_source_path(image, "图片")

    for camera, video in collect_named_sources(args, "video").items():
        if video:
            camera_videos[camera] = resolve_source_path(video, "视频")

    if not camera_images and not camera_videos:
        raise ValueError(
            "请至少指定一个模拟源：使用 --video、--image、--camera-video、--camera-image 或 --a-video 等参数。"
        )

    for camera, image_path in camera_images.items():
        set_source_env(camera, "image", image_path)

    for camera, video_path in camera_videos.items():
        set_source_env(camera, "video", video_path)

    import app as bike_app

    print("=" * 60)
    print("模拟摄像头已启用：")
    for camera, video_path in camera_videos.items():
        print(f"  {camera}: 视频流 {video_path}")
    for camera, image_path in camera_images.items():
        if camera not in camera_videos:
            print(f"  {camera}: 静态图片 {image_path}")
        else:
            print(f"  {camera}: 已设置图片 {image_path}，但视频优先生效")
    print(f"访问地址：http://{args.host}:{args.port}")
    print("=" * 60)

    bike_app.app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

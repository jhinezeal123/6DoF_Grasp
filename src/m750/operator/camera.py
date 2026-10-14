"""Process camera riêng; tái sử dụng capture/MjpegStream và luôn stop stream."""

import argparse
import threading


def main(argv=None):
    parser = argparse.ArgumentParser(description="Camera của menu ./start")
    parser.add_argument("mode", choices=("stream", "photo"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--out", default=".local_data/camera.jpg")
    args = parser.parse_args(argv)
    from m750.camera import MjpegStream, capture

    if args.mode == "photo":
        return 0 if capture(path=args.out, device=args.device) else 1
    stream = MjpegStream(device=args.device, port=args.port)
    try:
        stream.start()
        threading.Event().wait()
    except KeyboardInterrupt:
        return 130
    finally:
        stream.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

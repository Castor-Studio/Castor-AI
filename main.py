import argparse

def main():
    parser = argparse.ArgumentParser(description="Castor AI launcher")

    parser.add_argument(
        "--mode",
        choices=["foot", "handball", "podcast"],
        required=True,
        help="Mode d'exécution"
    )

    # --- Arguments video ---
    parser.add_argument("--video", type=str, help="Chemin vidéo unique (handball OU foot test)")
    parser.add_argument("--video-left", type=str, help="Chemin vidéo gauche (foot dual)")
    parser.add_argument("--video-right", type=str, help="Chemin vidéo droite (foot dual)")
    parser.add_argument("--frameskip", type=int, default=0, help="Frameskip")

    args = parser.parse_args()

    # =========================
    # FOOT MODE (dual OU mono test)
    # =========================
    if args.mode == "foot":
        from modules.foot_ai.inference import run as run_foot

        # Cas mono : --video
        if args.video and (args.video_left or args.video_right):
            parser.error("En mode foot, utilise soit --video (mono), soit --video-left + --video-right (dual), pas un mix.")

        if args.video:
            # Foot mono (test tracking)
            run_foot(args.video, None, frameskip=args.frameskip)
            return

        # Cas dual : --video-left + --video-right
        if not args.video_left or not args.video_right:
            parser.error("En mode foot, fournir soit --video (mono), soit le duo --video-left et --video-right (dual).")

        run_foot(args.video_left, args.video_right, frameskip=args.frameskip)

    # =========================
    # HANDBALL MODE (mono vidéo)
    # =========================
    elif args.mode == "handball":
        from modules.handball_ai.inference import run as run_hand

        if not args.video:
            parser.error("--video est requis en mode handball")

        run_hand(args.video, frameskip=args.frameskip)

    # =========================
    # PODCAST MODE
    # =========================
    elif args.mode == "podcast":
        print("Mode podcast non encore implémenté.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
Interactive COLOR rollout for SmolVLA + RTC.

Controls
--------
1 : Pick a blue ball and put it on the hand
2 : Pick a pink ball and put it on the hand
3 : Pick a yellow ball and put it on the hand
4 : Pick a purple ball and put it on the hand

0 / s : STOP current task and wait
q     : Quit

Important
---------
- SmolVLA policy is loaded only once.
- Robot and cameras are connected only once.
- Switching colors does NOT reload the checkpoint.
- STOP clears the RTC action queue and pauses execution.
- Original LeRobot source files are not modified.
"""

import logging
import select
import sys
import termios
import time
import tty

# Import the original rollout module so LeRobot registers
# the same robot/camera/config types as normal `lerobot-rollout`.
import lerobot.scripts.lerobot_rollout as _original_rollout  # noqa: F401

from lerobot.configs import parser
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import (
    init_visualization,
    shutdown_visualization,
)

logger = logging.getLogger(__name__)


# ============================================================
# COLOR TASKS
# ============================================================

COLOR_TASKS = {
    "1": "Pick a blue ball and put it on the hand",
    "2": "Pick a pink ball and put it on the hand",
    "3": "Pick a yellow ball and put it on the hand",
    "4": "Pick a purple ball and put it on the hand",
}

COLOR_NAMES = {
    "1": "BLUE",
    "2": "PINK",
    "3": "YELLOW",
    "4": "PURPLE",
}


# ============================================================
# MENU
# ============================================================

def print_color_menu():
    print()
    print("=" * 60)
    print("          SmolVLA COLOR TASK SELECTOR")
    print("=" * 60)
    print()
    print("   1  : BLUE")
    print("   2  : PINK")
    print("   3  : YELLOW")
    print("   4  : PURPLE")
    print()
    print("   0  : STOP / WAIT")
    print("   s  : STOP / WAIT")
    print()
    print("   q  : QUIT")
    print()
    print("=" * 60)
    print(" Enter 없이 키 하나만 누르면 됩니다.")
    print("=" * 60)
    print()


# ============================================================
# COLOR BASE STRATEGY
# ============================================================

class ColorBaseStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)

        self._active_task = None

        # Original terminal configuration.
        self._old_terminal_settings = None


    # ========================================================
    # SETUP
    # ========================================================

    def setup(self, ctx):
        """
        Initialize the normal LeRobot inference engine
        and put terminal into single-key input mode.
        """

        super().setup(ctx)

        if not isinstance(self._engine, RTCInferenceEngine):
            raise RuntimeError(
                "COLOR rollout은 RTC inference 전용입니다.\n"
                "--inference.type=rtc 를 사용하세요."
            )

        # ----------------------------------------------------
        # Single-key terminal input
        # ----------------------------------------------------

        if sys.stdin.isatty():
            self._old_terminal_settings = termios.tcgetattr(
                sys.stdin.fileno()
            )

            # Enter 없이 키 하나씩 읽기.
            tty.setcbreak(sys.stdin.fileno())

        print_color_menu()

        logger.info("COLOR rollout ready")
        logger.info("Waiting for color selection...")


    # ========================================================
    # KEYBOARD
    # ========================================================

    def _poll_keyboard(self):
        """
        Non-blocking keyboard polling.

        Returns:
            "1", "2", "3", "4"
            "0" or "s"
            "q"
            None
        """

        if not sys.stdin.isatty():
            return None

        ready, _, _ = select.select(
            [sys.stdin],
            [],
            [],
            0,
        )

        if not ready:
            return None

        key = sys.stdin.read(1).lower()

        if key in (
            "1",
            "2",
            "3",
            "4",
            "0",
            "s",
            "q",
        ):
            return key

        return None


    # ========================================================
    # RTC THREAD STOP HELPER
    # ========================================================

    def _stop_rtc_thread(self):
        """
        Stop current RTC inference completely.

        This does NOT unload SmolVLA.
        It only stops the RTC worker thread.
        """

        engine = self._engine

        engine.pause()

        old_thread = getattr(
            engine,
            "_rtc_thread",
            None,
        )

        engine.stop()

        # RTC stop() itself has a timeout.
        # For safe task switching, ensure inference
        # really finished before changing state.
        if (
            old_thread is not None
            and old_thread.is_alive()
        ):
            logger.info(
                "Waiting for current RTC inference to finish..."
            )

            old_thread.join()


    # ========================================================
    # STOP CURRENT TASK
    # ========================================================

    def _stop_current_task(self):
        """
        Stop current task.

        Policy remains loaded.
        Robot/cameras remain connected.

        Clears:
        - RTC inference thread
        - action queue
        - policy state
        - processor state
        - interpolation state

        After STOP, robot receives no new policy actions.
        """

        print()
        print("=" * 60)
        print("                  TASK STOPPED")
        print("=" * 60)
        print()
        print("SmolVLA 모델은 그대로 로드되어 있습니다.")
        print("로봇 / 카메라도 연결된 상태입니다.")
        print()
        print("다음 task 선택:")
        print("1=BLUE  2=PINK  3=YELLOW  4=PURPLE")
        print()
        print("=" * 60)
        print()

        logger.info("Stopping current COLOR task")

        # Stop running RTC inference.
        self._stop_rtc_thread()

        # Clear RTC/policy state.
        self._engine.reset()

        # Clear any remaining interpolated action.
        self._interpolator.reset()

        # Force fresh observation when new task starts.
        self._cached_obs_processed = None

        self._active_task = None

        logger.info(
            "COLOR task stopped. Policy remains loaded."
        )


    # ========================================================
    # SWITCH COLOR TASK
    # ========================================================

    def _switch_task(self, command):
        """
        Change the language instruction without
        reloading SmolVLA.
        """

        engine = self._engine
        interpolator = self._interpolator

        new_task = COLOR_TASKS[command]
        color_name = COLOR_NAMES[command]

        print()
        print("=" * 60)
        print(f"              TASK -> {color_name}")
        print("=" * 60)
        print()
        print(new_task)
        print()
        print("=" * 60)
        print()

        logger.info(
            "Switching RTC task to: %s",
            new_task,
        )

        # ----------------------------------------------------
        # Stop previous task inference.
        # ----------------------------------------------------

        self._stop_rtc_thread()

        # ----------------------------------------------------
        # Change ONLY language instruction.
        #
        # SmolVLA checkpoint/model object remains in memory.
        # ----------------------------------------------------

        engine._task = new_task

        # ----------------------------------------------------
        # Clear all previous task state.
        # ----------------------------------------------------

        engine.reset()

        interpolator.reset()

        self._cached_obs_processed = None

        # ----------------------------------------------------
        # Start a fresh RTC worker.
        #
        # This does NOT run from_pretrained().
        # Therefore model isn't reloaded.
        # ----------------------------------------------------

        engine.start()

        engine.resume()

        self._active_task = command

        logger.info(
            "RTC task switch complete: %s",
            new_task,
        )

        print()
        print(
            f"[RUNNING] {color_name}"
        )
        print()
        print(
            "STOP: 0 또는 s"
        )
        print(
            "다른 색으로 바로 변경: 1 / 2 / 3 / 4"
        )
        print(
            "종료: q"
        )
        print()


    # ========================================================
    # MAIN CONTROL LOOP
    # ========================================================

    def run(self, ctx):

        cfg = ctx.runtime.cfg

        robot = ctx.hardware.robot_wrapper

        interpolator = self._interpolator

        control_interval = (
            interpolator.get_control_interval(
                cfg.fps
            )
        )

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Normal BaseStrategy calls engine.resume() here.
        #
        # COLOR rollout intentionally does NOT.
        #
        # Robot waits until 1/2/3/4 is pressed.
        # ----------------------------------------------------

        logger.info(
            "Control loop started."
        )

        logger.info(
            "Press 1/2/3/4 to start."
        )

        while not ctx.runtime.shutdown_event.is_set():

            loop_start = time.perf_counter()


            # =================================================
            # KEYBOARD
            # =================================================

            command = self._poll_keyboard()


            # -------------------------------------------------
            # QUIT
            # -------------------------------------------------

            if command == "q":

                print()
                print()
                print("QUIT requested.")
                print()

                logger.info(
                    "Quit requested"
                )

                break


            # -------------------------------------------------
            # STOP
            # -------------------------------------------------

            if command in (
                "0",
                "s",
            ):

                if self._active_task is not None:

                    self._stop_current_task()

                else:

                    print()
                    print(
                        "[WAITING] 이미 STOP 상태입니다."
                    )
                    print()

                # Don't run robot control during this tick.
                continue


            # -------------------------------------------------
            # COLOR CHANGE
            # -------------------------------------------------

            if command in COLOR_TASKS:

                self._switch_task(
                    command
                )

                # Let the new RTC thread acquire
                # a fresh observation first.
                continue


            # =================================================
            # DURATION
            # =================================================

            if cfg.duration > 0:

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                if elapsed >= cfg.duration:

                    logger.info(
                        "Duration limit reached (%.0fs)",
                        cfg.duration,
                    )

                    break


            # =================================================
            # WAIT MODE
            #
            # No task selected or STOP was pressed.
            # =================================================

            if self._active_task is None:

                dt = (
                    time.perf_counter()
                    - loop_start
                )

                sleep_t = (
                    control_interval
                    - dt
                )

                if sleep_t > 0:
                    precise_sleep(
                        sleep_t
                    )

                continue


            # =================================================
            # NORMAL LEROBOT CONTROL LOOP
            # =================================================

            obs = robot.get_observation()


            obs_processed = (
                self._process_observation_and_notify(
                    ctx.processors,
                    obs,
                )
            )


            # -------------------------------------------------
            # torch.compile warmup
            # -------------------------------------------------

            if self._handle_warmup(
                cfg.use_torch_compile,
                loop_start,
                control_interval,
            ):
                continue


            # -------------------------------------------------
            # Get RTC action and send to SO101
            # -------------------------------------------------

            action_dict = send_next_action(
                obs_processed,
                obs,
                ctx,
                interpolator,
            )


            # -------------------------------------------------
            # Visualization telemetry
            # -------------------------------------------------

            self._log_telemetry(
                obs_processed,
                action_dict,
                ctx.runtime,
            )


            # -------------------------------------------------
            # Maintain control FPS
            # -------------------------------------------------

            dt = (
                time.perf_counter()
                - loop_start
            )

            sleep_t = (
                control_interval
                - dt
            )

            if sleep_t > 0:

                precise_sleep(
                    sleep_t
                )

            else:

                if dt > 0:

                    logger.warning(
                        "Control loop slower than target: "
                        "%.1f Hz < %.1f Hz",
                        1 / dt,
                        cfg.fps,
                    )


    # ========================================================
    # TEARDOWN
    # ========================================================

    def teardown(self, ctx):

        try:

            # ------------------------------------------------
            # Stop RTC first.
            # ------------------------------------------------

            if self._engine is not None:

                self._stop_rtc_thread()


            # ------------------------------------------------
            # Normal BaseStrategy teardown:
            #
            # - stop inference
            # - optionally return initial pose
            # - disconnect robot
            # - disconnect cameras
            # ------------------------------------------------

            super().teardown(ctx)


        finally:

            # ------------------------------------------------
            # Restore terminal mode.
            # ------------------------------------------------

            if (
                self._old_terminal_settings
                is not None
                and sys.stdin.isatty()
            ):

                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    self._old_terminal_settings,
                )

                self._old_terminal_settings = None


# ============================================================
# COLOR ROLLOUT ENTRY POINT
# ============================================================

@parser.wrap()
def color_rollout(cfg: RolloutConfig):

    init_logging()


    # --------------------------------------------------------
    # COLOR rollout only supports base + RTC.
    # --------------------------------------------------------

    if cfg.strategy.type != "base":

        raise ValueError(
            "COLOR rollout은 "
            "--strategy.type=base 전용입니다."
        )


    if cfg.inference.type != "rtc":

        raise ValueError(
            "COLOR rollout은 "
            "--inference.type=rtc 전용입니다."
        )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    if cfg.display_data:

        logger.info(
            "Initializing %s visualization "
            "(ip=%s, port=%s)",
            cfg.display_mode,
            cfg.display_ip,
            cfg.display_port,
        )

        init_visualization(
            cfg.display_mode,
            session_name="color_rollout",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )


    # --------------------------------------------------------
    # Shutdown handler
    # --------------------------------------------------------

    signal_handler = (
        ProcessSignalHandler(
            use_threads=True,
            display_pid=False,
        )
    )

    shutdown_event = (
        signal_handler.shutdown_event
    )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # This loads:
    #
    # - SmolVLA
    # - processors
    # - SO101
    # - cameras
    # - RTC inference engine
    #
    # ONLY ONCE.
    # --------------------------------------------------------

    logger.info(
        "Building COLOR rollout context..."
    )

    logger.info(
        "SmolVLA policy will be loaded ONCE."
    )

    ctx = build_rollout_context(
        cfg,
        shutdown_event,
    )


    strategy = ColorBaseStrategy(
        cfg.strategy
    )


    logger.info(
        "Strategy: COLOR BASE"
    )

    logger.info(
        "Robot: %s | FPS: %.0f | Duration: %s",
        (
            cfg.robot.type
            if cfg.robot
            else "?"
        ),
        cfg.fps,
        (
            f"{cfg.duration}s"
            if cfg.duration > 0
            else "infinite"
        ),
    )


    try:

        strategy.setup(ctx)

        logger.info(
            "Setup complete."
        )

        logger.info(
            "1/2/3/4 = task, "
            "0/s = stop, "
            "q = quit"
        )

        strategy.run(ctx)


    except KeyboardInterrupt:

        logger.info(
            "Interrupted by user"
        )


    finally:

        strategy.teardown(ctx)

        if cfg.display_data:

            shutdown_visualization(
                cfg.display_mode
            )


    logger.info(
        "COLOR rollout finished"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    register_third_party_plugins()

    color_rollout()


if __name__ == "__main__":
    main()



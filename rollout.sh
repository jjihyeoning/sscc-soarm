#!/usr/bin/env bash

# 경로
SCRIPT="/home/sscc/rrobot/lerobot/src/lerobot/scripts/lerobot_color_rollout.py"
POLICY="/home/sscc/rrobot/lerobot/outputs/train/ball/smolvla100_sort_tune/checkpoints/040000/pretrained_model"

# 카메라
TOP=2
GRIPPER=0

# 표시
DISPLAY=false

# 로봇
ROBOT_PORT="/dev/ttyACM0"
ROBOT_ID="so101_follower_arm"




echo
echo "=========================================="
echo "Top camera     : $TOP"
echo "Gripper camera : $GRIPPER"
echo "Policy         : $POLICY"
echo "Display data   : $DISPLAY"
echo "Robot port     : $ROBOT_PORT"
echo "=========================================="
echo


python "$SCRIPT" \
  --strategy.type=base \
  --policy.path="$POLICY" \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="{top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30}, gripper: {type: opencv, index_or_path: $GRIPPER, width: 640, height: 480, fps: 30}}" \
  --task="Pick a pink ball and put it on the hand" \
  --duration=0 \
  --display_data="$DISPLAY" \
  --inference.type=rtc

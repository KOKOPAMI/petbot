def get_direction(center_x, frame_center, deadzone=50):
    """
    객체 중심 좌표 기준으로 방향 판단
    """

    if center_x < frame_center - deadzone:
        return "LEFT"
    
    elif center_x > frame_center + deadzone:
        return "RIGHT"
    
    else:
        return "CENTER"
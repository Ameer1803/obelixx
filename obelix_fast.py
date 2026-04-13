"""OBELIX variation: optimized for training speed by disabling rendering.

- No cv2.imshow, cv2.waitKey, or rendering overhead.
- Always runs in headless mode.
- Keeps all game logic intact for accurate training.
"""

from __future__ import annotations
import cv2
import numpy as np

from obelix import OBELIX


class OBELIXFast(OBELIX):
    def _update_frames(self, show: bool) -> None:
        # Skip rendering entirely for speed.
        # Still compute masks and frames for observation logic.
        self.frame = np.ones(self.frame_size, np.uint8) * 0
        self.bot_mask = np.ones(self.frame_size, np.uint8) * 0
        cv2.rectangle(
            self.frame,
            (0 + 5, 0 + 5),
            (self.frame_size[0] - 5, self.frame_size[1] - 5),
            (255, 0, 0),
            1,
        )
        cv2.rectangle(
            self.frame,
            (0 + 10, 0 + 10),
            (self.frame_size[0] - 10, self.frame_size[1] - 10),
            (255, 0, 0),
            1,
        )

        self.box_frame = np.zeros(self.frame_size, np.uint8)
        if self.box_visible or self.enable_push:
            self.box_corners = []
            for i in range(0, 360, 90):
                x = self.box_center_x + (self.box_size // 2) * np.cos(
                    np.deg2rad(self.box_yaw_angle + i)
                )
                y = self.box_center_y + (self.box_size // 2) * np.sin(
                    np.deg2rad(self.box_yaw_angle + i)
                )
                self.box_corners.append([x, y])
            cv2.fillPoly(
                self.box_frame,
                np.array([self.box_corners], dtype=np.int32),
                (100, 100, 100),
            )

        self.obstacle_frame = np.zeros(self.frame_size, np.uint8)
        for p1, p2 in self.obstacles:
            cv2.rectangle(self.obstacle_frame, p1, p2, (100, 100, 100), -1)

        self.sensor_feedback_masks = np.zeros(
            (9, self.frame_size[0], self.frame_size[1]), np.uint8
        )

        cv2.circle(
            self.frame,
            (self.bot_center_x, self.bot_center_y),
            self.bot_radius,
            self.bot_color,
            1,
        )
        cv2.circle(
            self.bot_mask,
            (self.bot_center_x, self.bot_center_y),
            self.bot_radius,
            (100, 100, 100),
            -1,
        )

        for sonar_range, sonar_intensity in zip(
            [self.sonar_far_range, self.sonar_near_range, self.sonar_range_offset],
            [100, 50, 0],
        ):
            for index, (sonar_pos_angle, sonar_face_angle) in enumerate(
                zip(self.sonar_positions, self.sonar_facing_angles)
            ):
                if sonar_intensity == 0:
                    noise_reduction = 2
                else:
                    noise_reduction = 0
                p1_x = self.bot_center_x + self.bot_radius * np.cos(
                    np.deg2rad(self.facing_angle + sonar_pos_angle)
                )
                p1_y = self.bot_center_y + self.bot_radius * np.sin(
                    np.deg2rad(self.facing_angle + sonar_pos_angle)
                )
                p2_x = p1_x + sonar_range * np.cos(
                    np.deg2rad(
                        self.facing_angle
                        + sonar_face_angle
                        + self.sonar_fov // 2
                        + noise_reduction
                    )
                )
                p2_y = p1_y + sonar_range * np.sin(
                    np.deg2rad(
                        self.facing_angle
                        + sonar_face_angle
                        + self.sonar_fov // 2
                        + noise_reduction
                    )
                )
                p3_x = p1_x + sonar_range * np.cos(
                    np.deg2rad(
                        self.facing_angle
                        + sonar_face_angle
                        - self.sonar_fov // 2
                        - noise_reduction
                    )
                )
                p3_y = p1_y + sonar_range * np.sin(
                    np.deg2rad(
                        self.facing_angle
                        + sonar_face_angle
                        - self.sonar_fov // 2
                        - noise_reduction
                    )
                )

                cv2.fillPoly(
                    self.frame,
                    np.array(
                        [[[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]]], dtype=np.int32
                    ),
                    sonar_intensity,
                )
                cv2.fillPoly(
                    self.sensor_feedback_masks[index],
                    np.array(
                        [[[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]]], dtype=np.int32
                    ),
                    sonar_intensity,
                )

        p1_x = int(
            self.bot_center_x + self.bot_radius * np.cos(np.deg2rad(self.facing_angle))
        )
        p1_y = int(
            self.bot_center_y + self.bot_radius * np.sin(np.deg2rad(self.facing_angle))
        )
        p2_x = int(p1_x + self.ir_sensor_range * np.cos(np.deg2rad(self.facing_angle)))
        p2_y = int(p1_y + self.ir_sensor_range * np.sin(np.deg2rad(self.facing_angle)))
        cv2.line(self.frame, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
        cv2.line(
            self.sensor_feedback_masks[8], (p1_x, p1_y), (p2_x, p2_y), (50, 50, 50), 2
        )

        self.frame = cv2.addWeighted(self.frame, 1.0, self.box_frame, 1.0, 0)
        self.frame = cv2.addWeighted(self.frame, 1.0, self.obstacle_frame, 1.0, 0)
        self.frame = cv2.addWeighted(self.frame, 1.0, self.neg_circle_frame, 1.0, 0)
        self.frame = cv2.flip(self.frame, 0)

        # Skip all cv2.imshow and rendering calls.

    def render_frame(self):
        # No-op for speed.
        pass

    def update_state_diagram(self):
        # No-op for speed.
        pass

    def step(self, move, render=True):
        # Force render=False to skip rendering overhead.
        return super().step(move, render=False)
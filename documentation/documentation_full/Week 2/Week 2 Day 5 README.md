# Quest Log Day 5 — Detailed Documentation

Today was another uneventful day in terms of lectures and seminars. We did not have a daily seminar, and our instructor did not teach or lecture us, so we had more time to prepare for our tests and work with the RACECAR. Even with the additional time, our testing and preparation did not go well. We made basically no successful progress with the line follower because the car could not reliably sense the line. We tried multiple controllers and value adjustments, but there was no completed run.

---

## Testing Plan

Our main expectation was to test the line follower using **bang-bang, PD, proportional, and single-setpoint proportional controllers**. We wanted to choose the controller that worked best and then continue adjusting and improving its values. We also tested wall-following code when the line-following problems prevented consistent progress.

---

## Testing Session 1

We completed **20 runs** during the first testing session. Every run resulted in a **DNF**. We first tested several versions of the line PD controller. The car hit the walls, went straight without reacting to the line, or oscillated before hitting the left wall. We considered several possible causes, including errors that were too large, a low turn angle, excessive Kp, and the line not being sensed.

We also tested multiple bang-bang versions. Some versions did not move because the maximum-speed setting needed to be adjusted. Other versions moved but hit a wall early or oscillated. One bang-bang run immediately turned into the right wall, and we did not know the cause at the time.

The proportional and single-setpoint proportional line-following versions also did not finish. The car often went straight and then hit the left wall, moved too slowly, stopped after adjusting, or failed to see the line. We changed the speed, steering angle, and setpoint behavior, but these changes did not produce a successful run.

We also completed one wall PD run. It reached a curve but hit the wall, indicating that the Kp and Kd values still needed adjustment for the selected speed.

The best run from this session was **Run 18, Line PD 7**. It had less oscillation than the earlier PD attempts but failed at the curve. Based on this result, we decided to continue fixing the PD controller.

---

## Testing Session 2

We completed another **15 runs** in the second testing session, and all of them also resulted in a **DNF**. Several PD runs went straight before hitting the left wall. The logs showed very large error values in some runs, while later tests suggested that the car was not consistently seeing the line. Slowing the car allowed it to travel farther in one run, but it still lost the line after the adjustments. This led us to consider the HSV range and camera detection as possible problems.

The proportional and single-setpoint proportional tests showed similar behavior. The car sometimes went straight, turned right, and then made a very large correction toward the left wall. Other runs overshot when starting at a curve or immediately hit the left wall because the steering angle was too high. During the final recorded run, repeatedly clicking Auto appeared to let the car see the line. This suggested that the setpoint or camera information might have been updating too slowly for the car to react while moving.

We also performed three wall-following tests. At low speeds, the car jittered and hit the left wall. We suspected a hardware issue because the gears appeared to be slipping. Another wall-following run went straight into the wall, which could have involved either the code or the sensor input.

---

## Data Collection Method

We tried using an **Excel sheet for data collection** instead of relying only on raw logs. For every run, we recorded the lab name, result, error situation, and a potential cause of the error. This allowed us to attach a short explanation to each test so that we would not have to sort through the entire log later.

However, this method also created problems. Reviewing the footage and manually reviewing and entering the data took too long. The recorded footage also took up too much space in my Google Drive. In addition, we tried to keep normal program logs and optimize the testing process, but the logs glitched and we lost the data we had collected. Therefore, although the Excel sheet made individual runs easier to compare, the complete process was too slow and required too much storage.

---

## Conclusion and Next Steps

All **35 recorded runs** ended in a **DNF**. The repeated failures did not appear to come only from poor controller values. Many tests suggested that the camera could not reliably sense the line, and possible hardware slipping also affected some wall-following runs. Because controller tuning is not reliable when the sensor input is inconsistent, the next step should be to verify the camera image, crop region, contour detection, HSV range, and setpoint update before continuing to adjust Kp, Kd, speed, or steering angle. After line detection is reliable, we will continue with the PD controller because Run 18 showed the best behavior during the first testing session.

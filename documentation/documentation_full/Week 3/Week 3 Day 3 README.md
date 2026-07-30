> Our goal for the day was to try our best to reach speed 2.5, although we expected that this would be difficult. We planned to continue testing and implementing our bang-bang wall-following controller while adding assisting techniques, such as two more side rays, to improve the car’s response. If we finished the Speed Quest early, we planned to return to the Quest Log and work on other algorithms.

---

## Testing and Results

We reached speed 1.5 and earned one coin. The bang-bang controller worked while we increased the speed, but around speed 1.5, the car started to hit walls. The controller had already oscillated at lower speeds because of the bang-bang method, but the oscillations became larger as the speed increased.

We considered switching to the PD controller that had already been written or using a different follow-the-gap method. However, testing occurred late in the day, so there was not enough time to properly compare both alternatives and decide which one to continue using. The disagreement over the controller choice also disrupted the team’s workflow. For the next testing period, we need a clearer comparison process using the bang-bang controller as the baseline and testing the PD and follow-the-gap methods under similar conditions.

We had hoped to return to the Quest Log, but the Speed Quest will continue into the next day. The new Quest Log speed requirements are also much lower, so our initial speed goals and testing plans will need to be adjusted.

---

## Technical Learning

Before the Draper visit, a Beaver Works research student discussed wall-following approaches other than PID control. Because a higher speed gives the controller less time to react between feedback updates, the instructor wanted us to begin at a low speed and increase it incrementally instead of immediately testing at a high speed. We continued using bang-bang as our main controller because it had worked well at lower speeds, while keeping PD and follow-the-gap as backup approaches.

The testing rotation also changed. Students were divided among three rooms, with one room used for official trials. The smaller preparation groups allowed us to test the cars with the provided tube walls before an official trial. The rotations were also increased from 15 minutes to 30 minutes, giving us more time to prepare, test, and adjust the controller.

---

## Next Steps

Continue the Speed Quest by comparing the PD and follow-the-gap controllers with the bang-bang baseline. Test each method at gradually increasing speeds, observe wall collisions and oscillation, and determine whether additional side rays improve stability. After completing the Speed Quest, return to the revised Quest Log goals.

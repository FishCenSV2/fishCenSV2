# ByteTrack
This documentation contains some info about the ByteTrack library. Much of the details are abstracted away since the tracker code only involves one function call and access to two member variables.

## Table of Contents
- [ByteTrack](#bytetrack)
  - [Table of Contents](#table-of-contents)
  - [Changes from Original](#changes-from-original)
  - [Code Explanations](#code-explanations)
    - [Max Time Lost](#max-time-lost)
    - [Output Tracks](#output-tracks)

## Changes from Original
There are two main changes from the original code
- Code to prevent memory leak on Lines 226 to 236 in [`BYTETracker.cpp`](https://github.com/FishCenSV2/fishCenSV2/blob/main/libs/bytetrack/BYTETracker.cpp)
- Change of member variable `vector<STrack> removed_stracks` from `private` to `public` in [`BYTETracker.h`](https://github.com/FishCenSV2/fishCenSV2/blob/main/libs/bytetrack/BYTETracker.h)

Both of these changes have to do with the member variable `vector<STrack> removed_stracks`. This vector contains objects that are no longer being tracked as they have exceeded the timeout specified by the tracker. This vector in the original code was never cleared which could cause a memory leak. The following lines were added to fix this

```cpp
/*
Fix to prevent memory leak. Unsure if this will negatively impact
performance. Based on how Ultralytics dealt with it.

NOTE: Technically our main function already takes care of this but this
  is kept here for consistency sakes. The library shouldn't rely on the
  user to take care of this outside of the class.
*/
if(this->removed_stracks.size() > 1000) {
  this->removed_stracks.erase(this->removed_stracks.begin(),this->removed_stracks.begin()+500);
}
```

The next change of `vector<STrack> removed_stracks` from `private` to `public` was needed since the code in `main.cpp` needs to know the objects that aren't being tracked anymore and remove them from the previous positions map object. Otherwise, we would need to keep track of them on our own which is just reinventing the wheel since the tracker already does this.

There is one more change that has no impact on the code but fixed an annoying issue. In `dataType.h` the following lines were added

```cpp
//This is only needed since Intellisense keeps putting namespace errors.
#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif
```

IntelliSense in VSCode was being annoying and kept yelling about namespace errors even though everything compiled fine. This is specifically due to the eigen3 library and the fix was from [here](https://github.com/microsoft/vscode-cpptools/issues/7413).

## Code Explanations
This section attempts to explain a few details that would have overcrowded the `main.cpp` code overview.


### Max Time Lost
---
The tracker is initialized in the following way

```cpp
constexpr int fps = 50;

BYTETracker tracker(fps,30); 
```

The `fps` argument and the second argument actually determine the `max_time_lost` member variable which is assigned in the `BYTETracker` class's constructor

```cpp
max_time_lost = int(frame_rate / 30.0 * track_buffer);
```

This member variable is then only used once in the whole class. This is lines 189-197 of the `update` method 

```cpp
////////////////// Step 5: Update state //////////////////
for (int i = 0; i < this->lost_stracks.size(); i++)
{
  if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
  {
  	this->lost_stracks[i].mark_removed();
  	removed_stracks.push_back(this->lost_stracks[i]);
  }
}
```

I am unsure how this calculation exactly works so if anyone else has possible ideas let me know. For now it might suffice to just play around with the arguments of the constructor to increase this time if needed.

### Output Tracks
When we update the tracker with the bounding box detections we get a vector of `STrack` objects

```cpp
std::vector<Object> objects;                //Vector of bounding boxes
std::vector<STrack> output_stracks;         //Vector of tracked objects

//Other setup code here...

constexpr int fps = 50;

BYTETracker tracker(fps,30); 

//Bunch of code for inference here...

output_stracks = tracker.update(objects);
```

We note that each element in `output_stracks` is directly related to an element in `objects`. For example `output_stracks[0]` refers to the same bounding box as `objects[0]`. 

For the `STrack` object we only care about several member variables: `track_id`, `tlbr`, and `tlwh`. The `track_id` is just the unique ID given to each bounding box. Meanwhile, `tlbr` and `tlwh` are vectors of size 4 which define the coordinates/shape of the bounding box. For `tlbr` it stands for "Top left bottom right" which means `tlbr[0]` and `tlbr[1]` are the xy coordinates for the top left of the box while `tlbr[1]` and `tlbr[2]` are the xy coordinates for the bottom right of the box. 

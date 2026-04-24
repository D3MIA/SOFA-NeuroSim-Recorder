import re, os
d = 'simulation_output/run_E2.50_nu0.450_seed1111/images'
n = 2000
nums = sorted([
    int(m.group(1))
    for fn in os.listdir(d)
    if (m := re.match(r'frame_(\d+)\.(jpg|jpeg|png)$', fn, re.IGNORECASE))
    and int(m.group(1)) < n
])
print(f"Images within NPZ range (frame < {n}): {len(nums)}")
print(f"First 5: {nums[:5]}")
print(f"Last 5:  {nums[-5:]}")
gaps = [nums[i+1]-nums[i] for i in range(min(10, len(nums)-1))]
print(f"Gaps between first 10 pairs: {gaps}")
print(f"Detected image_every: {round(sum(gaps)/len(gaps))}")

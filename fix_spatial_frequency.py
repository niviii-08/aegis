with open('src/image/models/spatial_frequency.py', 'r') as f:
    lines = f.readlines()

# Find the problematic section and fix it
fixed_lines = []
skip_until = -1

for i, line in enumerate(lines):
    if i == 193 and '@dataclass(frozen=True)' in line:
        # Skip this incorrectly placed decorator
        continue
    elif i == 194 and 'class SpatialFrequencyConfig:' in line:
        # Add the decorator before this line
        fixed_lines.append('@dataclass(frozen=True)\n')
        fixed_lines.append('class SpatialFrequencyConfig:\n')
        # Skip the next few lines until we find the proper class definition
        skip_until = 200
    elif i == 195 and 'class SpatialFrequencyModel(nn.Module):' in line:
        # This is the real class definition that should come after the dataclass
        if skip_until > 0:
            skip_until = -1
            fixed_lines.append(line)
    elif skip_until > 0 and i < skip_until:
        # Skip lines in the problematic section
        continue
    else:
        fixed_lines.append(line)

with open('src/image/models/spatial_frequency_fixed.py', 'w') as f:
    f.writelines(fixed_lines)

print('Fixed file created as src/image/models/spatial_frequency_fixed.py')
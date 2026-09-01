with open('src/image/models/spatial_frequency.py', 'r') as f:
    lines = f.readlines()

# Find where to insert the missing FrequencyBranchConfig
fixed_lines = lines[:183]  # Keep everything up to the end of FrequencyBranch class
fixed_lines.append('\n')  # Add spacing
fixed_lines.append('@dataclass(frozen=True)\n')  # Add decorator
fixed_lines.append('class FrequencyBranchConfig:\n')  # Add class definition
fixed_lines.append('    """Configuration for the frequency-domain (FFT) branch."""\n')  # Add docstring
fixed_lines.append('    enabled: bool = True\n')  # Add field
fixed_lines.append('    out_features: int = 128\n')  # Add field
fixed_lines.append('    dropout: float = 0.2\n')  # Add field
fixed_lines.append('\n')  # Add spacing
fixed_lines.append('\n')  # Add another spacing line
fixed_lines.extend(lines[183:])  # Add the rest of the file

with open('src/image/models/spatial_frequency_fixed.py', 'w') as f:
    f.writelines(fixed_lines)

print('Added missing FrequencyBranchConfig')
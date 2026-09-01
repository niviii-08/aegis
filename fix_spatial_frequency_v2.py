with open('src/image/models/spatial_frequency.py', 'r') as f:
    lines = f.readlines()

# Properly reconstruct the file
fixed_lines = lines[:183]  # Keep everything up to line 183
fixed_lines.append('@dataclass(frozen=True)\n')  # Add the decorator
fixed_lines.append('class SpatialFrequencyConfig:\n')  # Add the class definition
fixed_lines.append('    spatial_branch: SpatialBranchConfig = field(default_factory=SpatialBranchConfig)\n')  # Add field
fixed_lines.append('    frequency_branch: FrequencyBranchConfig = field(default_factory=FrequencyBranchConfig)\n')  # Add field
fixed_lines.append('    fusion_dropout: float = 0.2\n')  # Add field
fixed_lines.append('\n')  # Add proper spacing
fixed_lines.append('\n')  # Add another spacing line
fixed_lines.append('class SpatialFrequencyModel(nn.Module):\n')  # Add the model class definition
fixed_lines.append('    """Spatial + Frequency fusion model for binary real/fake classification.\n')  # Add docstring start
fixed_lines.append('\n')  # Add spacing
fixed_lines.append('    Can operate in three modes:\n')  # Add docstring line
fixed_lines.append('    - Spatial only:  frequency_branch disabled\n')  # Add docstring line
fixed_lines.append('    - Frequency only: spatial_branch disabled\n')  # Add docstring line
fixed_lines.append('    - Fusion:         both branches enabled (features concatenated)\n')  # Add docstring line
fixed_lines.append('    """\n')  # Add docstring end
fixed_lines.append('\n')  # Add spacing
fixed_lines.extend(lines[200:])  # Add the rest of the file starting from __init__

with open('src/image/models/spatial_frequency_fixed.py', 'w') as f:
    f.writelines(fixed_lines)

print('Fixed file created as src/image/models/spatial_frequency_fixed.py')
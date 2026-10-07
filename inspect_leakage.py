import sys
sys.path.insert(0, 'src')
from image.splits.leakage_checker import check_splits
import inspect
print(inspect.signature(check_splits))

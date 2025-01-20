# Troubleshooting

## Error: Missing Dependency
**Description**: When running the framework, you may encounter an error like:
```
No valid project found in the current directory
```

**Cause**: This happens when a required library is not installed.

**Solution**:
1. Ensure all dependencies are installed by running:
   ```bash
   pip install tensorflow
   pip install numpy
   pip install opencv-python
   pip install tensorboard

**Note**: These libraries are required for the first run of your project.

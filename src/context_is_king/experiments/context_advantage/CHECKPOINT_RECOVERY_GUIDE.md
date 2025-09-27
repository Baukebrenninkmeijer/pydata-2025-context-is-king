# Context Advantage Experiment - Checkpoint & Recovery Guide

## Overview

The Context Advantage Experiment now supports comprehensive checkpointing and graceful recovery mechanisms, allowing experiments to be interrupted and resumed without losing progress. The system uses a dual-checkpoint approach:
1. **Results Checkpoints**: Saved after EVERY experiment run as JSON lists
2. **Full Checkpoints**: Saved periodically with complete state tracking

## Key Features

### 🔄 Automatic Checkpointing

#### Results Checkpoints (After EVERY run)
- **Immediate saving**: Results saved as JSON list after each experiment completes
- **Separate files**: Needle and LongMemEval results stored separately
- **Simple format**: List of dictionaries for easy inspection and processing
- **Location**: `output_dir/checkpoints/[experiment_id]_[needle|longmem]_results.json`

#### Full State Checkpoints (Periodic)
- **Periodic saves**: Complete state saved every 5 minutes (configurable)
- **Atomic writes**: Uses temporary files to prevent corruption
- **Progress tracking**: Saves completed tasks and remaining work
- **State preservation**: Maintains full experiment configuration and results

### 🛡️ Graceful Shutdown
- **Signal handling**: Responds to SIGINT (Ctrl+C) and SIGTERM
- **Safe interruption**: Completes current task before stopping
- **State saving**: Automatically saves checkpoint on interruption
- **Resume instructions**: Provides command to resume experiment

### 📂 Recovery Mechanisms
- **Experiment restoration**: Resumes from exact interruption point
- **Result preservation**: Maintains all completed results
- **Task continuation**: Picks up remaining tasks seamlessly
- **Phase awareness**: Continues from correct experiment phase

## Command Line Usage

### List Available Checkpoints
```bash
python context_advantage_experiment.py --list-checkpoints
```

### Resume from Checkpoint
```bash
python context_advantage_experiment.py --resume-from EXPERIMENT_ID
```

### Configure Checkpoint Interval
```bash
python context_advantage_experiment.py --checkpoint-interval 600  # Save every 10 minutes
```

### Full Example with Recovery
```bash
# Start experiment
python context_advantage_experiment.py --experiment-type both --models claude-sonnet-4 claude-haiku-3.5

# If interrupted, resume with:
python context_advantage_experiment.py --resume-from abc12345
```

## Checkpoint Structure

### Results Checkpoints (JSON Lists)
Stored in `output_dir/checkpoints/`:

**Needle Results** (`[experiment_id]_needle_results.json`):
```json
[
  {
    "experiment_id": "abc12345",
    "model_name": "claude-haiku-3.5",
    "context_size": 10000,
    "composition": "mixed",
    "retrieval_scores": [0.9, 0.85, 0.92],
    "success_rate": 0.95,
    ...
  },
  ...
]
```

**LongMemEval Results** (`[experiment_id]_longmem_results.json`):
```json
[
  {
    "experiment_id": "abc12345",
    "model_name": "claude-sonnet-4",
    "context_group": "long",
    "question_scores": [0.8, 0.75, 0.9],
    "success_rate": 0.85,
    ...
  },
  ...
]
```

**Summary** (`[experiment_id]_summary.json`):
```json
{
  "experiment_id": "abc12345",
  "timestamp": 1703097600.0,
  "needle_results_count": 10,
  "longmem_results_count": 5,
  "config": { ... }
}
```

### Full State Checkpoints
Stored in `data/checkpoints/context_advantage/`:

```json
{
  "experiment_id": "abc12345",
  "config": { ... },
  "completed_needle_results": [...],
  "completed_longmem_results": [...],
  "remaining_needle_tasks": [...],
  "remaining_longmem_tasks": [...],
  "current_phase": "needle|longmemeval|complete",
  "completed_tasks": 5,
  "total_tasks": 20,
  "timestamp": 1703097600.0,
  "last_saved": 1703097600.0
}
```

## Recovery Scenarios

### 1. Normal Interruption (Ctrl+C)
- Experiment catches signal
- Completes current task
- Saves checkpoint
- Displays resume command

### 2. Crash/Kill
- Previous checkpoint available
- Resume from last saved state
- Some progress may be lost (max 5 minutes)

### 3. Planned Pause
- Use Ctrl+C to gracefully stop
- Resume later with provided command
- No data loss

## Phase-Aware Recovery

The system tracks experiment phases:
- **needle**: Running needle-in-haystack experiments
- **needle_interrupted**: Needle phase was interrupted
- **longmemeval**: Running LongMemEval experiments  
- **longmemeval_interrupted**: LongMemEval phase was interrupted
- **complete**: All experiments finished

Recovery automatically continues from the appropriate phase.

## Advanced Usage

### Programmatic API
```python
from context_advantage_experiment import ContextAdvantageExperiment

# Initialize experiment
experiment = ContextAdvantageExperiment()

# Save results checkpoint (after each run)
experiment.save_results_checkpoint(config)

# Load results checkpoint
needle_count, longmem_count = experiment.load_results_checkpoint(config)

# List full checkpoints
checkpoints = experiment.list_checkpoints()

# Load specific full checkpoint
checkpoint = experiment.load_checkpoint('experiment_id')

# Save full checkpoint manually
experiment.save_checkpoint(config, remaining_tasks=tasks)

# Clean up completed experiment
experiment.cleanup_checkpoint('experiment_id')
```

### Custom Checkpoint Intervals
```python
experiment = ContextAdvantageExperiment()
experiment.checkpoint_interval = 300  # 5 minutes
```

## Troubleshooting

### Checkpoint Not Found
- Check `data/checkpoints/context_advantage/` directory
- Verify experiment ID is correct
- Use `--list-checkpoints` to see available checkpoints

### Permission Errors
- Ensure write permissions on data directory
- Check disk space availability

### Corrupted Checkpoint
- Delete corrupted checkpoint file
- Restart experiment from beginning
- Consider shorter checkpoint intervals

## Best Practices

1. **Monitor disk space**: Checkpoints can be large for big experiments
2. **Regular cleanup**: Remove old checkpoints after successful completion
3. **Backup important results**: Copy final results to safe location
4. **Test recovery**: Verify checkpoint/recovery works before long experiments
5. **Use appropriate intervals**: Balance between safety and performance

## Technical Implementation

### Signal Handling
```python
signal.signal(signal.SIGINT, self._signal_handler)
signal.signal(signal.SIGTERM, self._signal_handler)
```

### Atomic Checkpoint Saving
```python
temp_path = checkpoint_path.with_suffix('.tmp')
with open(temp_path, 'w') as f:
    json.dump(checkpoint_data, f, indent=2)
temp_path.rename(checkpoint_path)  # Atomic operation
```

### Task State Management
- Tracks completed vs remaining tasks
- Preserves experiment configuration
- Maintains result collections
- Records current execution phase

This robust checkpoint and recovery system ensures that long-running context advantage experiments can be safely interrupted and resumed, protecting valuable computational work and API costs.
# Always Winning Plinko - Encoder Calibration Tool

This calibration script maps physical catcher positions to encoder counts, allowing the main Plinko code to accurately position the catcher.

## Setup

1. Flash `always-winning-plinko-calibration.ino` to your Arduino
2. Open the Serial Monitor at **9600 baud**

## Quick Start Workflow

1. **Zero the encoder** at a known reference position:
   - Manually position the catcher at your "zero" point( or use `<` and `>` commands to move it)
   - Type `z` and press Enter

2. **Move the catcher** to calibration positions:
   - Use `>` for a small step forward
   - Use `<` for a small step reverse
   - Use `F` for continuous forward, `R` for continuous reverse
   - Use `X` to stop continuous motion

3. **Record samples** by typing the physical position value:
   - Move catcher to a marked position (e.g., 5 cm from zero)
   - Type `5` and press Enter
   - The script records the current encoder count paired with that position

4. **Repeat** for multiple positions (at least 2-3 recommended)

5. **Save calibration** to EEPROM:
   - Type `w` and press Enter

## Command Reference

| Command | Action |
|---------|--------|
| `z` | Zero encoder (resets count to 0) |
| `<number>` | Record sample at current encoder count for given physical position |
| `p` | Toggle continuous encoder count printing |
| `c` | Clear all collected samples |
| `s` | Show current calibration coefficients |
| `w` | Write calibration to EEPROM |
| `q` | Read calibration from EEPROM |
| `d` | Delete/clear EEPROM calibration |
| `>` | Step motor forward (short pulse) |
| `<` | Step motor reverse (short pulse) |
| `F` | Continuous forward (auto-stops after 60s) |
| `R` | Continuous reverse (auto-stops after 60s) |
| `X` | Stop motor immediately |
| `h` | Show help |

## Example Session

```
z                  # Zero at left edge
> > > >            # Step forward to the 10cm mark
10                 # Record: position 10 → current count
> > > > >          # Step forward to 25cm mark
25                 # Record: position 25 → current count
s                  # View calibration (counts = a*pos + b)
w                  # Save to EEPROM
```

## Output

After recording samples, the script computes a linear fit: `counts = a × position + b`

- **a** = encoder counts per unit of physical distance
- **b** = offset (ideally near 0 if you zeroed properly)
- **RMSE** = root-mean-square error of the fit

## Tips

- Use consistent units (cm, inches, etc.) for all position entries
- Take samples across the full range of catcher travel
- More samples = better calibration accuracy
- The motor has a 60-second safety timeout for continuous motion
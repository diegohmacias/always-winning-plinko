// Motor-assisted encoder calibration tool
// Use low PWM and short pulses to move the catcher and record encoder counts.
// Commands over Serial (see help below).

#include <EEPROM.h>

const int ENA = 5;   // PWM enable
const int IN1 = 7;
const int IN2 = 8;

const int encoderPinA = 2;
const int encoderPinB = 3;

volatile long encoderPos = 0;
volatile int lastEncoded = 0;

const int MAX_SAMPLES = 24;
double samplePos[MAX_SAMPLES];
double sampleCountArr[MAX_SAMPLES];
int sampleCountStored = 0;

bool printContinuous = false;

// calibration counts = a * pos + b
double a_coeff = 0.0;
double b_coeff = 0.0;

struct CalEEPROM { float a; float b; };
const int EEPROM_ADDR = 0;

// Motor drive defaults (safe)
const int DEFAULT_PWM = 110;   // 0..255 (lower is safer)
const unsigned long STEP_MS = 120;  // ms for small step
const unsigned long CONTINUOUS_SAFE_MS = 60000UL; // 60s safety timeout

// motor state
bool motorRunning = false;
unsigned long motorStartMillis = 0;
int motorDir = 0; // 0 stopped, +1 forward, -1 reverse
int motorPWM = DEFAULT_PWM;

void updateEncoder() {
  int MSB = digitalRead(encoderPinA);
  int LSB = digitalRead(encoderPinB);
  int encoded = (MSB << 1) | LSB;
  int sum = (lastEncoded << 2) | encoded;
  if (sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderPos++;
  if (sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderPos--;
  lastEncoded = encoded;
}

void setup() {
  Serial.begin(9600);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  digitalWrite(ENA, LOW);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, CHANGE);

  Serial.println(F("=== Motor-assisted Encoder Calibration ==="));
  printHelp();
  readCalibrationFromEEPROM();
  delay(200);
}

void loop() {
  // enforce motor safety timeout
  if (motorRunning && (millis() - motorStartMillis > CONTINUOUS_SAFE_MS)) {
    stopMotor();
    Serial.println(F("Motor automatic STOP (safety timeout)"));
  }

  if (Serial.available() > 0) {
    String in = Serial.readStringUntil('\n');
    in.trim();
    if (in.length() == 0) return;
    // single-letter commands:
    if (in.equalsIgnoreCase("z")) {
      zeroEncoder();
    } else if (in.equalsIgnoreCase("p")) {
      toggleContinuousPrint();
    } else if (in.equalsIgnoreCase("c")) {
      clearSamples();
    } else if (in.equalsIgnoreCase("s")) {
      computeAndShowCalibration();
    } else if (in.equalsIgnoreCase("w")) {
      writeCalibrationToEEPROM();
    } else if (in.equalsIgnoreCase("q")) {
      readCalibrationFromEEPROM();
    } else if (in.equalsIgnoreCase("d")) {
      deleteCalibrationFromEEPROM();
    } else if (in.equalsIgnoreCase(">")) {
      stepForward();
    } else if (in.equalsIgnoreCase("<")) {
      stepReverse();
    } else if (in.equalsIgnoreCase("F")) {
      startContinuousForward();
    } else if (in.equalsIgnoreCase("R")) {
      startContinuousReverse();
    } else if (in.equalsIgnoreCase("X")) {
      stopMotor();
    } else if (in.equalsIgnoreCase("h")) {
      printHelp();
    } else {
      // numeric input: record sample physical position
      double pos = in.toDouble();
      if (pos == 0.0 && in != "0" && in != "0.0") {
        Serial.println(F("Unrecognized command. Type 'h' for help."));
      } else {
        addSample(pos);
      }
    }
  }

  if (printContinuous) {
    noInterrupts();
    long c = encoderPos;
    interrupts();
    Serial.print("Count: ");
    Serial.println(c);
    delay(200);
  }
}

// ---------- motor helpers ----------
void driveMotor(int dir, int pwmVal) {
  // dir: +1 forward (IN1=HIGH, IN2=LOW), -1 reverse, 0 stop (brake)
  if (dir == 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0);
    motorRunning = false;
    motorDir = 0;
    return;
  }
  motorDir = dir;
  motorRunning = true;
  motorStartMillis = millis();
  if (dir > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  }
  analogWrite(ENA, pwmVal);
}

void stopMotor() {
  // set ENA LOW for disable (may brake depending on driver)
  analogWrite(ENA, 0); // ensures no PWM
  // optionally set both IN low to brake (common)
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  motorRunning = false;
  motorDir = 0;
  Serial.println(F("Motor STOPPED"));
}

void stepForward() {
  Serial.println(F("STEP forward (short pulse)"));
  driveMotor(+1, motorPWM);
  delay(STEP_MS);
  stopMotor();
}

void stepReverse() {
  Serial.println(F("STEP reverse (short pulse)"));
  driveMotor(-1, motorPWM);
  delay(STEP_MS);
  stopMotor();
}

void startContinuousForward() {
  Serial.println(F("Start continuous forward (will auto-stop after safety timeout)"));
  driveMotor(+1, motorPWM);
}

void startContinuousReverse() {
  Serial.println(F("Start continuous reverse (will auto-stop after safety timeout)"));
  driveMotor(-1, motorPWM);
}

// ---------- encoder & sampling ----------
void zeroEncoder() {
  noInterrupts();
  encoderPos = 0;
  interrupts();
  Serial.println(F(">> Encoder ZEROED"));
}

void addSample(double physPos) {
  noInterrupts();
  long c = encoderPos;
  interrupts();
  if (sampleCountStored >= MAX_SAMPLES) {
    Serial.println(F("ERROR: sample buffer full. Use 'c' to clear."));
    return;
  }
  samplePos[sampleCountStored] = physPos;
  sampleCountArr[sampleCountStored] = (double)c;
  sampleCountStored++;
  Serial.print(F("Sample added: pos="));
  Serial.print(physPos, 6);
  Serial.print(F("  count="));
  Serial.println(c);
  computeAndShowCalibration();
}

// compute least-squares fit counts = a * pos + b
void computeAndShowCalibration() {
  if (sampleCountStored < 1) {
    Serial.println(F("No samples."));
    return;
  }
  if (sampleCountStored == 1) {
    double x = samplePos[0];
    double y = sampleCountArr[0];
    if (fabs(x) < 1e-9) {
      Serial.println(F("Single sample at pos=0 -> can't compute slope"));
      return;
    }
    a_coeff = y / x;
    b_coeff = 0.0;
  } else {
    double Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for (int i = 0; i < sampleCountStored; ++i) {
      double x = samplePos[i];
      double y = sampleCountArr[i];
      Sx += x; Sy += y; Sxx += x * x; Sxy += x * y;
    }
    double n = sampleCountStored;
    double denom = (n * Sxx - Sx * Sx);
    if (fabs(denom) < 1e-12) {
      Serial.println(F("Degenerate sample set (all positions equal)"));
      return;
    }
    a_coeff = (n * Sxy - Sx * Sy) / denom;
    b_coeff = (Sy - a_coeff * Sx) / n;
  }
  Serial.println(F("--- CALIBRATION (counts = a*pos + b) ---"));
  Serial.print(F("samples: "));
  Serial.println(sampleCountStored);
  Serial.print(F("a (counts per unit): "));
  Serial.println(a_coeff, 9);
  Serial.print(F("b (count offset): "));
  Serial.println(b_coeff, 9);
  if (fabs(a_coeff) > 1e-12) {
    double units_per_count = 1.0 / a_coeff;
    Serial.print(F("units_per_count: "));
    Serial.println(units_per_count, 12);
  }
  double ss = 0;
  for (int i = 0; i < sampleCountStored; ++i) {
    double pred = a_coeff * samplePos[i] + b_coeff;
    double err = sampleCountArr[i] - pred;
    ss += err * err;
  }
  double rmse = sqrt(ss / (double)sampleCountStored);
  Serial.print(F("RMSE (counts): "));
  Serial.println(rmse, 6);
  Serial.println(F("----------------------------------------"));
}

void writeCalibrationToEEPROM() {
  CalEEPROM cal; cal.a = (float)a_coeff; cal.b = (float)b_coeff;
  EEPROM.put(EEPROM_ADDR, cal);
  Serial.println(F("Calibration written to EEPROM."));
}

void readCalibrationFromEEPROM() {
  CalEEPROM cal; EEPROM.get(EEPROM_ADDR, cal);
  a_coeff = (double)cal.a; b_coeff = (double)cal.b;
  Serial.println(F("Calibration read from EEPROM (may be zeros)."));
  Serial.print(F("a=")); Serial.println(a_coeff, 9);
  Serial.print(F("b=")); Serial.println(b_coeff, 9);
}

void deleteCalibrationFromEEPROM() {
  CalEEPROM cal; cal.a = 0.0f; cal.b = 0.0f;
  EEPROM.put(EEPROM_ADDR, cal);
  a_coeff = 0.0; b_coeff = 0.0;
  Serial.println(F("EEPROM calibration cleared."));
}

void toggleContinuousPrint() {
  printContinuous = !printContinuous;
  Serial.print(F("Continuous printing: "));
  Serial.println(printContinuous ? "ON" : "OFF");
}

void clearSamples() {
  sampleCountStored = 0;
  Serial.println(F("Samples CLEARED"));
}

void printHelp() {
  Serial.println(F("Commands:"));
  Serial.println(F("z -> zero encoder (set counts = 0)"));
  Serial.println(F("<num>-> record calibration sample: enter physical position (units you choose)"));
  Serial.println(F("p -> toggle continuous encoder count printing"));
  Serial.println(F("c -> clear collected samples"));
  Serial.println(F("s -> compute & show calibration (a,b)"));
  Serial.println(F("w -> write calibration (a,b) to EEPROM"));
  Serial.println(F("q -> read calibration from EEPROM"));
  Serial.println(F("d -> delete EEPROM calibration"));
  Serial.println(F("> -> step forward (short pulse)"));
  Serial.println(F("< -> step reverse (short pulse)"));
  Serial.println(F("F -> start continuous forward (low speed, auto-stop after timeout)"));
  Serial.println(F("R -> start continuous reverse"));
  Serial.println(F("X -> stop motor"));
  Serial.println(F("h -> help"));
  Serial.println(F(""));
  Serial.println(F("Procedure: mark physical positions on catcher, use > or F to move to a mark, then type the physical position number and Enter to record."));
  Serial.println(F("Use small steps (>) to get repeatable alignment. Keep PWM low to avoid high currents."));
}


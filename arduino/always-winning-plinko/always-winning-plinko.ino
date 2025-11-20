// ------------------------------------------------------
// Arduino Sketch — Position Control: Catcher Carriage Drive
// Motor control pins: ENA=5 (PWM), IN1=7, IN2=8
// Encoder pins: A=2, B=3
// Motor+Encoder: Pololu 37D (64 CPR)
// Serial input: from Pi, ball_x_norm (0.0 .. 1.0) on newline
// This script performs the following:
// 1. Powers up and performs "Homing" (finds zero)
// 2. Listens for x_norm (0.0-1.0) from Pi
// 3. Converts x_norm -> inches -> encoder counts
// 4. Uses PID to drive carriage to target
// ------------------------------------------------------

#include <PID_v1.h>

// --- PINS ---
const int ENA = 5;  // PWM
const int IN1 = 7;
const int IN2 = 8;
const int encoderPinA = 2;
const int encoderPinB = 3;

// --- USER CONFIGURATION (UPDATE THESE AFTER CALIBRATION) ---
// 1. How many encoder counts equal 1 inch (or 1 cm)?
//    Calculate this from your manual calibration data (Slope).
float COUNTS_PER_INCH = 500.0; 

// 2. What is the physical width of the board (in same units as above)?
//    This maps x_norm 1.0 to a physical distance.
float BOARD_WIDTH_INCHES = 20.0; 

// 3. PID Tuning Parameters (IMPORTANT TO TUNE)
double Kp = 2.0, Ki = 0.5, Kd = 1.0; 
// ----------------------------------------------------------

volatile long encoderPos = 0;
volatile int lastEncoded = 0;

// PID Variables
double setpoint = 0.0;
double inputVal = 0.0;
double outputVal = 0.0;

PID myPID(&inputVal, &outputVal, &setpoint, Kp, Ki, Kd, DIRECT);

// Encoder Interrupt Routine
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
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);
  
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, CHANGE);

  // --- HOMING ROUTINE ---
  Serial.println("HOMING: Moving Left to find Zero...");
  
  // Drive LEFT slowly to hit the hard stop
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH); 
  analogWrite(ENA, 80); // low speed (approx 30%)

  // Run for 2 seconds (adjust depending on board length)
  // This ensures we hit the wall even if we started in the middle.
  delay(2000); 
  
  // stop motor
  analogWrite(ENA, 0); 
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  
  // reset encoder (this is now physical 0)
  noInterrupts();
  encoderPos = 0; 
  lastEncoded = 0;
  interrupts();
  
  Serial.println("HOMED. Encoder reset to 0.");
  // ----------------------

  myPID.SetMode(AUTOMATIC);
  myPID.SetOutputLimits(-255, 255);
  Serial.println("Ready for commands.");
}

void loop() {
  // 1. RECEIVE DATA
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    double x_norm = data.toFloat(); // Value between 0.0 and 1.0

    // 2. CONVERT TO PHYSICAL TARGET
    // distance in inches
    double targetInches = x_norm * BOARD_WIDTH_INCHES;
    
    // convert to counts (calibration Math)
    double targetCounts = targetInches * COUNTS_PER_INCH;

    // 3. SOFT LIMITS (safety)
    // prevent trying to drive past the board width
    double maxAllowedCounts = BOARD_WIDTH_INCHES * COUNTS_PER_INCH;
    
    if (targetCounts < 0) targetCounts = 0;
    if (targetCounts > maxAllowedCounts) targetCounts = maxAllowedCounts;

    setpoint = targetCounts;

    Serial.print("Norm: "); Serial.print(x_norm);
    Serial.print(" -> Inch: "); Serial.print(targetInches);
    Serial.print(" -> Setpoint: "); Serial.println(setpoint);
  }

  // 4. PID CONTROL
  inputVal = (double)encoderPos;
  myPID.Compute();

  // 5. MOTOR DRIVE
  if (outputVal > 0) {
    // drive right
    analogWrite(ENA, (int)outputVal);
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else if (outputVal < 0) {
    // drive left
    analogWrite(ENA, (int)(-outputVal));
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  } else {
    // stop
    analogWrite(ENA, 0);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
  }

  // debug output every 200ms
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint >= 200) {
    // only print if moving or significant error to keep serial quiet
    if (abs(setpoint - inputVal) > 50) {
        Serial.print("Pos: "); Serial.print(encoderPos);
        Serial.print(" / Tgt: "); Serial.print(setpoint);
        Serial.print(" / PWM: "); Serial.println(outputVal);
    }
    lastPrint = millis();
  }
}

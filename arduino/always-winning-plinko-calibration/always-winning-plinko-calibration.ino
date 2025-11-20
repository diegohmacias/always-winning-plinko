// ------------------------------------------------------
// MANUAL CALIBRATION SKETCH
// 1. Upload this sketch.
// 2. Open Serial Monitor (Baud 9600).
// 3. Move carriage by hand to known positions.
// 4. Type the distance (e.g. "10") and press Enter.
// ------------------------------------------------------

const int ENA = 5;
const int IN1 = 7;
const int IN2 = 8;

const int encoderPinA = 2;
const int encoderPinB = 3;

volatile long encoderPos = 0;
volatile int lastEncoded = 0;

// interrupt service routine
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

  // motor pins setup
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // DISABLE MOTOR for manual movement
  digitalWrite(ENA, LOW); 
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  // encoder Setup
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, CHANGE);

  Serial.println("--- MANUAL CALIBRATION MODE ---");
  Serial.println("1. Push carriage to your Zero point (e.g. Left Edge).");
  Serial.println("2. Type 'z' and Enter to reset encoder to 0.");
  Serial.println("3. Move carriage to a known distance (e.g. 1.0 in).");
  Serial.println("4. Type that distance (e.g. '1') and Enter to log it.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim(); // remove any whitespace/newlines

    // command to zero the encoder
    if (input.equalsIgnoreCase("z")) {
      noInterrupts(); // pause interrupts to write safely
      encoderPos = 0;
      interrupts(); // resume interrupts
      Serial.println(">> Encoder ZEROED <<");
    } 
    // otherwise, treat input as a physical measurement label
    else {
      // tead safely into a local variable
      noInterrupts();
      long currentCount = encoderPos;
      interrupts();

      Serial.print("Physical_Pos: ");
      Serial.print(input);
      Serial.print("\t | Encoder_Count: ");
      Serial.println(currentCount);
    }
  }
}

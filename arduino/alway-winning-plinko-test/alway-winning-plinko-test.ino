// Simple Echo Test
// This script is meant to test the Raspberry Pi 4 to Arduino Uno R3
// serial communication. 

void setup() {
  Serial.begin(9600); // match this baud rate on Pi side
  while (!Serial) {
    ; // wait for serial port to connect
  }
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');

    // echo back what was received
    Serial.print("Received: ");
    Serial.println(data);
  }
}

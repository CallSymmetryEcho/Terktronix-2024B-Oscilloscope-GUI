const int CHANNELS[] = {A0, A1, A2, A3};  // 模拟输入引脚
const int SAMPLE_INTERVAL = 50;       // 采样间隔(ms)
bool isRunning = false;               // 运行状态标志

void setup() {
  Serial.begin(115200);  // 高波特率
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "start") {
      isRunning = true;
      Serial.println("Recording started");
    } 
    else if (command == "stop") {
      isRunning = false;
      Serial.println("Recording stopped");
    }
  }

  if (isRunning) {
    for (int i = 0; i < sizeof(CHANNELS)/sizeof(CHANNELS[0]); i++) {
      int value = analogRead(CHANNELS[i]);
      float voltage = value * (5.0 / 1023.0);
      Serial.print(voltage, 4);
      Serial.print(",");
    }
    Serial.println();
    delay(SAMPLE_INTERVAL);
  }
}
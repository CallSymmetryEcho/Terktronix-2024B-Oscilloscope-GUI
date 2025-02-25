const int CHANNELS[] = {A0, A1, A2};  // 模拟输入引脚
const int SAMPLE_INTERVAL = 50;       // 采样间隔(ms)

void setup() {
  Serial.begin(115200);  // 高波特率
  // 使用默认的5V参考电压
}

void loop() {
  for (int i = 0; i < sizeof(CHANNELS)/sizeof(CHANNELS[0]); i++) {
    int value = analogRead(CHANNELS[i]);
    float voltage = value * (5.0 / 1023.0);  // 转换为实际电压值
    Serial.print(voltage, 3);  // 输出3位小数的电压值
    Serial.print(",");
  }
  Serial.println();
  delay(SAMPLE_INTERVAL);
}
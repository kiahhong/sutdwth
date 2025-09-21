#include <WiFi.h>
#include <HTTPClient.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include <SPI.h>
#include <ArduinoJson.h>
#include <math.h>

// ===== TFT Setup =====
#define TFT_CS   7
#define TFT_DC   16
#define TFT_RST  18
Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);

#define ECG_COLOR     ILI9341_GREEN
#define BG_COLOR      ILI9341_BLACK
#define TEXT_COLOR    ILI9341_WHITE
#define ACCENT_COLOR  ILI9341_CYAN
#define ALERT_COLOR   ILI9341_RED
#define INFO_COLOR    ILI9341_YELLOW
#define DARKGREY      0x7BEF
int screenW, screenH;



// ===== WiFi =====
#define WIFI_SSID "SUTD_Guest"
#define WIFI_PASSWORD ""

// ===== InfluxDB (SEND) =====
#define INFLUXDB_URL_SEND   ""
#define INFLUXDB_TOKEN_SEND ""
#define INFLUXDB_ORG_SEND   "project_1"
#define INFLUXDB_BUCKET_SEND "test_1"

// ===== InfluxDB (FETCH) =====
#define INFLUXDB_URL_FETCH   "
#define INFLUXDB_TOKEN_FETCH ""
#define INFLUXDB_ORG_FETCH   "project_1"
#define INFLUXDB_BUCKET_FETCH "prediction"

// ===== ECG =====
#define ECG_PIN 5
#define SAMPLE_RATE_HZ 20
#define RECORD_SECONDS 5
#define BATCH_SECONDS 1

const int TOTAL_SAMPLES = SAMPLE_RATE_HZ * RECORD_SECONDS;
const int BATCH_SAMPLES = SAMPLE_RATE_HZ * BATCH_SECONDS;
int ecg_data[TOTAL_SAMPLES];
bool dataSent = false;

// ===== Button =====
const int buttonPin = 4;
int buttonState = HIGH;
int lastButtonState = HIGH;

// ===== MP3 Pins =====
int songPins[] = {35, 36, 37, 38, 39};
int numSongs = sizeof(songPins) / sizeof(songPins[0]);
int songIndex = 0; // next song to play

unsigned long lastUpdate = 0;
const unsigned long updateInterval = 4000; // 4 seconds
String lastPrediction = "";

int ecgPattern[] = {
  120,120,121,123,126,130,
  100,80,60,140,180,150,
  120,122,124,128,126,124,
  122,120,120,120,120
};
int ecgPatternLen = sizeof(ecgPattern)/sizeof(ecgPattern[0]);

void drawCenteredText(const char *msg, int offsetY, uint16_t color=TEXT_COLOR, int size=2) {
  int16_t x = (screenW - strlen(msg)*6*size)/2;
  int16_t y = (screenH/2) + offsetY;
  tft.setTextColor(color);
  tft.setTextSize(size);
  tft.setCursor(x,y);
  tft.print(msg);
}

// ===== Setup =====
void setup() {
  Serial.begin(115200);

  // TFT init
  tft.begin();
  tft.setRotation(1);
  screenW = tft.width();
  screenH = tft.height();
  tft.fillScreen(BG_COLOR);

  // Pins
  pinMode(ECG_PIN, INPUT);
  pinMode(buttonPin, INPUT_PULLUP);

  // Initialize MP3 pins
  for (int i = 0; i < numSongs; i++) {
    pinMode(songPins[i], OUTPUT);
    digitalWrite(songPins[i], HIGH); // inactive
  }

  // WiFi
  connectWiFi();

  startupAnimation();
  playSong(1);       // fast power-on
  ecgIntroAnimation();       // realistic ECG sweep
  showProductName(); 
  playSong(1);
  pressToStartAnimation();  
}

void loop() {


  buttonState = digitalRead(buttonPin);
  if (buttonState == LOW && lastButtonState == HIGH && !dataSent) {
    Serial.println("Button pressed. Recording ECG...");
    playSong(2);
    instructionAnimation();
    playSong(4);
    delay(1200);
    playSong(3); 
    recordAndSendECG();
    dataSent = true;
    playSong(0); 

    recordingAnimation();
    analysingAnimation();
    showResult();
    playSong(1);  // <-- fetch + show latest prediction
  }
  lastButtonState = buttonState;
  delay(50);
}

// ===== WiFi =====
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" Connected!");
}

// ===== InfluxDB (Send) =====
bool sendToInflux(String data) {
  HTTPClient http;
  String url = String(INFLUXDB_URL_SEND) + "/api/v2/write?org=" + INFLUXDB_ORG_SEND + "&bucket=" + INFLUXDB_BUCKET_SEND + "&precision=s";
  http.begin(url);
  http.addHeader("Authorization", "Token " + String(INFLUXDB_TOKEN_SEND));
  http.addHeader("Content-Type", "text/plain; charset=utf-8");
  int response = http.POST(data);
  if (response == 204) { http.end(); return true; }
  else { 
    Serial.print("HTTP Error: "); Serial.println(response);
    Serial.print("Response: "); Serial.println(http.getString());
    http.end(); 
    return false; 
  }
}

String fetchPrediction() {
  HTTPClient http;
  String query = String("from(bucket: \"") + INFLUXDB_BUCKET_FETCH + "\")"
                 + " |> range(start: -12h)"
                 + " |> filter(fn: (r) => r._measurement == \"prediction\" and r._field == \"predicted_value\")"
                 + " |> last()";

  String url = String(INFLUXDB_URL_FETCH) + "/api/v2/query?org=" + INFLUXDB_ORG_FETCH;

  http.begin(url);
  http.addHeader("Authorization", "Token " + String(INFLUXDB_TOKEN_FETCH));
  http.addHeader("Content-Type", "application/vnd.flux");
  int response = http.POST(query);

  String resultValue = "Results in Database";
  if (response == 200) {
    String payload = http.getString();
    Serial.println("Query response: " + payload);

    // Split payload into lines
    int lineStart = 0;
    int lastCommaIndex;
    String lastValidLine = "";
    while (lineStart < payload.length()) {
      int lineEnd = payload.indexOf('\n', lineStart);
      if (lineEnd == -1) lineEnd = payload.length();
      String line = payload.substring(lineStart, lineEnd);
      line.trim();
      if (line.length() > 0 && line[0] != '#') {
        lastValidLine = line; // keep last non-header line
      }
      lineStart = lineEnd + 1;
    }

    // Split last valid line by commas
    int from = 0;
    int fieldIndex = 0;
    String fieldValue = "";
    for (int i = 0; i <= lastValidLine.length(); i++) {
      if (i == lastValidLine.length() || lastValidLine[i] == ',') {
        if (fieldIndex == 6) { // _value column
          fieldValue = lastValidLine.substring(from, i);
          fieldValue.trim();
          break;
        }
        from = i + 1;
        fieldIndex++;
      }
    }

    if (fieldValue.length() > 0) resultValue = fieldValue;

  } else {
    Serial.print("Query Error: ");
    Serial.println(response);
    Serial.println(http.getString());
  }

  http.end();
  return resultValue;
}


// ===== ECG Recording =====
void recordAndSendECG() {
  for (int i = 0; i < TOTAL_SAMPLES; i++) {
    ecg_data[i] = analogRead(ECG_PIN);
    delay(1000 / SAMPLE_RATE_HZ);
  }
  Serial.println("ECG Recording Complete.");

  for (int start = 0; start < TOTAL_SAMPLES; start += BATCH_SAMPLES) {
    int batch_count = min(BATCH_SAMPLES, TOTAL_SAMPLES - start);
    String postData = "";
    for (int i = 0; i < batch_count; i++) {
      postData += "ecg_data adc_value=" + String(ecg_data[start + i]) + "\n";
    }
    if (sendToInflux(postData)) {
      Serial.print("Batch "); Serial.print(start); Serial.println(" sent.");
    } else {
      Serial.print("Batch "); Serial.print(start); Serial.println(" failed.");
    }
    delay(10);
  }
}

// ===== MP3 playback =====
void playSong(int index) {
  if (index < 0 || index >= numSongs) {
    Serial.println("Invalid song index");
    return;
  }

  digitalWrite(songPins[index], LOW);
  delay(500);
  digitalWrite(songPins[index], HIGH);

  Serial.print("Triggered song index: ");
  Serial.println(index + 1);
}

// ---------- Startup ----------
void startupAnimation() {
  tft.fillScreen(BG_COLOR);
  const char *name = "HeartGuard";
  int cx = screenW / 2;
  int cy = screenH / 2;

  tft.setTextColor(ACCENT_COLOR);
  tft.setTextSize(4);
  int16_t x = cx - (strlen(name) * 12);
  int16_t y = cy - 16;
  tft.setCursor(x, y);
  tft.print(name);

  delay(800);
}

// ---------- ECG Intro ----------
void ecgIntroAnimation() {
  tft.fillScreen(BG_COLOR);
  int centerY = screenH / 2;
  int amplitude = screenH / 4;

  unsigned long start = millis();
  int prevY = centerY;
  for (int x = 0; x < screenW; x++) {
    int idx = (x * 2) % ecgPatternLen;
    int p = ecgPattern[idx];
    int y = centerY - ((p - 120) * amplitude / 60);
    tft.drawLine(x, prevY, x+1, y, ECG_COLOR);
    prevY = y;
    delay(2); // ~1.5s total
  }
  delay(300);
}

// ---------- Product Name ----------
void showProductName() {
  tft.fillScreen(BG_COLOR);
  const char *name = "HeartGuard";
  int cx = screenW / 2;
  int cy = screenH / 2;

  tft.setTextColor(TEXT_COLOR);
  tft.setTextSize(4);
  int16_t x = cx - (strlen(name) * 12);
  int16_t y = cy - 16;
  tft.setCursor(x, y);
  tft.print(name);

  delay(1500);
}

// ---------- Press to Start ----------
void pressToStartAnimation() {
  const char *line = "Press to Start Recording";
  unsigned long start = millis();

  while (millis() - start < 3000) {
    tft.fillScreen(BG_COLOR);
    tft.setTextColor(TEXT_COLOR);
    tft.setTextSize(2);
    int16_t x = (screenW - strlen(line)*12) / 2;
    int16_t y = screenH/2 - 10;
    tft.setCursor(x, y);
    tft.print(line);

    // flashing bar
    if ((millis()/400) % 2 == 0) {
      tft.fillRect(x, y+30, strlen(line)*12, 6, ACCENT_COLOR);
    }
    delay(100);
  }
}

// ---------- Instruction ----------
void instructionAnimation() {
  tft.fillScreen(BG_COLOR);
  drawCenteredText("Please apply ECG pads", -70, INFO_COLOR, 2);
  drawCenteredText("at appropriate areas", -40, INFO_COLOR, 2);
  drawCenteredText("Legs should not", 0, ALERT_COLOR, 2);
  drawCenteredText("touch the ground", 30, ALERT_COLOR, 2);
  delay(5000); // 5s
}

// ---------- Recording ----------
void recordingAnimation() {
  // Header text
  tft.fillScreen(BG_COLOR);
  drawCenteredText("ECG is being recorded", -40, ACCENT_COLOR, 2);
  drawCenteredText("Do not move", -10, INFO_COLOR, 2);

  // ECG simulation for 10s
  int baseY = screenH - 40; 
  int prevY = baseY;
  unsigned long ecgStart = millis();
  const unsigned long ecgDuration = 10000; // 10 seconds
  int x = 0;
  int patIdx = 0;

  // Clear ECG area
  tft.fillRect(0, baseY - 40, screenW, 50, BG_COLOR);

  while (millis() - ecgStart < ecgDuration) {
    // Clear current column
    tft.drawLine(x, baseY - 40, x, baseY + 10, BG_COLOR);

    // Simulated ECG sample
    int p = ecgPattern[patIdx % ecgPatternLen];
    int y = baseY - ((p - 120) * 30 / 60);

    // Draw new segment
    tft.drawLine(x, prevY, x + 1, y, ECG_COLOR);
    prevY = y;

    x++;
    if (x >= screenW) {
      x = 0;
      prevY = baseY;
      tft.fillRect(0, baseY - 40, screenW, 50, BG_COLOR);
    }
    patIdx++;
    delay(15);
  }

  delay(300);
}

// ---------- Analysing ----------
void analysingAnimation() {
  tft.fillScreen(BG_COLOR);
  int cx = screenW/2;
  int cy = screenH/2;
  int radius = 40;

  tft.setTextColor(TEXT_COLOR);
  drawCenteredText("Analysing", -radius-30, TEXT_COLOR, 2);

  int steps = 180;
  int totalMs = 12000;
  int stepDelay = totalMs/steps;

  for (int i=0; i<=steps; i++) {
    float pct = (float)i/steps;
    int angle = pct*360;

    // clear circle area
    tft.fillCircle(cx, cy, radius, BG_COLOR);

    // draw arc
    for (int a=0; a<angle; a+=4) {
      float rad = a * PI/180;
      int x = cx + cos(rad)*radius;
      int y = cy + sin(rad)*radius;
      tft.drawPixel(x,y, ACCENT_COLOR);
    }

    // percent text
    int percent = pct*100;
    String pstr = String(percent)+"%";
    tft.setTextSize(2);
    int16_t x = (screenW - pstr.length()*12)/2;
    int16_t y = cy - 8;
    tft.setCursor(x,y);
    tft.fillRect(x-5, y-5, pstr.length()*12+10, 20, BG_COLOR); // clear
    tft.print(pstr);

    delay(stepDelay);
  }
}


void showResult() {
  String prediction = fetchPrediction();
  
  tft.fillScreen(BG_COLOR);
  tft.setTextSize(2);
  tft.setTextColor(ILI9341_CYAN);
  tft.setCursor(40, screenH / 2 - 20);
  tft.println("Prediction:");
  tft.setCursor(40, screenH / 2 + 10);
  tft.setTextColor(ILI9341_YELLOW);
  tft.println(prediction);

  Serial.println("Displayed Prediction: " + prediction);
}



#pragma once
// #define POS1
// #define NO_MODEL

constexpr int kSearchIterations = 30000;

constexpr int kBoardSize = 5;

constexpr int kStartingFences = 3;

constexpr int kMaxMoves = 50;

constexpr int p1_start_first = 0;
constexpr int p1_start_second = kBoardSize / 2;
constexpr int p2_start_first = kBoardSize - 1;
constexpr int p2_start_second = kBoardSize / 2;

constexpr int kRandomMovesCount = 4;
constexpr int kGameCount = 200;

constexpr bool kPlayerInput = false;

constexpr int kMaxFileSize = 20000;

constexpr int kPathResilienceIters = 25;
constexpr int kPathResilienceWeight = 2;
constexpr int kResilienceFeatureLength = 10;

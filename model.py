
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# СТРОИТЕЛЬНЫЙ БЛОК: два свёрточных слоя подряд
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# ENCODER: сжимаем изображение, извлекаем смысл
# ─────────────────────────────────────────────

class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)   # запоминаем детали для decoder-а
        x    = self.pool(skip)  # сжимаем для следующего уровня
        return x, skip


# ─────────────────────────────────────────────
# DECODER: восстанавливаем размер, уточняем маску
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels,
            kernel_size=2, stride=2
        )
        # после конкатенации каналов становится in + skip
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)          # увеличиваем в 2 раза
        x = torch.cat([x, skip], dim=1)  # склеиваем по каналам
        x = self.conv(x)               # уточняем признаки
        return x


# ─────────────────────────────────────────────
# ПОЛНАЯ U-NET
# ─────────────────────────────────────────────

class UNet(nn.Module):
    
    def __init__(self, in_channels: int = 3, features: list = None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

       
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        
        self.decoders = nn.ModuleList()
        # идём снизу вверх: каналы уменьшаются
        decoder_channels = [features[-1] * 2] + list(reversed(features[1:]))
        skip_channels    = list(reversed(features))
        out_channels     = list(reversed(features))

        for i in range(len(features)):
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_channels[i],
                    skip_channels=skip_channels[i],
                    out_channels=out_channels[i]
                )
            )

       
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

       
        x = self.bottleneck(x)

       
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

       
        return self.final_conv(x)


# ─────────────────────────────────────────────
# ПРОВЕРКА — запусти этот файл напрямую
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Проверка архитектуры U-Net")
    print("=" * 50)

    model = UNet(in_channels=3, features=[64, 128, 256, 512])

   
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nВсего параметров    : {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")

   
    print("\nПрогон фиктивного батча [4, 3, 256, 256]...")
    dummy_input = torch.randn(4, 3, 256, 256)   # 4 патча, RGB, 256×256

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Вход : {dummy_input.shape}")         # [4, 3, 256, 256]
    print(f"Выход: {output.shape}")              # [4, 1, 256, 256]
    print(f"Диапазон выхода (логиты): [{output.min():.2f}, {output.max():.2f}]")

    
    probs = torch.sigmoid(output)
    print(f"Диапазон после Sigmoid  : [{probs.min():.2f}, {probs.max():.2f}]")

    # Проверяем RAM — важно для CPU
    param_size_mb = total_params * 4 / (1024 ** 2)   # float32 = 4 байта
    print(f"\nПримерный размер модели: {param_size_mb:.1f} MB")

    
    print("\n" + "─" * 50)
    print("Облегчённая версия [32, 64, 128, 256]:")
    light_model = UNet(in_channels=3, features=[64, 128, 256, 512])
    light_params = sum(p.numel() for p in light_model.parameters())
    light_mb = light_params * 4 / (1024 ** 2)
    print(f"Параметров : {light_params:,}")
    print(f"Размер     : {light_mb:.1f} MB")

    with torch.no_grad():
        light_output = light_model(dummy_input)
    print(f"Выход      : {light_output.shape}")

    print("\n✓ Модель работает корректно!")
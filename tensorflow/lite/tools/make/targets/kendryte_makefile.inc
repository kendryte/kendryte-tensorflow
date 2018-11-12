# Settings for Kendryte platforms.
ifeq ($(TARGET), kendryte)
  TARGET_ARCH := rv64imafdc
  TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-

  CXXFLAGS += \
  -ffunction-sections \
  -fdata-sections \
  -mcmodel=medany \
  -march=rv64imafdc \
  -fno-common
  CCFLAGS += \
  -ffunction-sections \
  -fdata-sections \
  -mcmodel=medany \
  -march=rv64imafdc \
  -fno-common
	LIBS += -ldl
endif

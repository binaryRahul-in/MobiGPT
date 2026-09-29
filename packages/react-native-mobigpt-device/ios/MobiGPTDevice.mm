// Adapted from PocketPal AI (MIT, (c) Asghar Ghorbani) — see NOTICE.md.
#import "MobiGPTDevice.h"

#import <Metal/Metal.h>
#import <UIKit/UIKit.h>
#include <mach/mach.h>
#include <os/proc.h>
#include <sys/sysctl.h>

static NSString *SysctlString(const char *name) {
  size_t size = 0;
  if (sysctlbyname(name, NULL, &size, NULL, 0) != 0 || size == 0) return @"";
  char *buf = (char *)malloc(size);
  sysctlbyname(name, buf, &size, NULL, 0);
  NSString *s = [NSString stringWithUTF8String:buf] ?: @"";
  free(buf);
  return s;
}

static BOOL SysctlFlag(const char *name) {
  int v = 0;
  size_t size = sizeof(v);
  return sysctlbyname(name, &v, &size, NULL, 0) == 0 && v != 0;
}

@implementation MobiGPTDevice

RCT_EXPORT_MODULE(HardwareInfo)

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

RCT_EXPORT_METHOD(getCPUInfo : (RCTPromiseResolveBlock)resolve reject : (RCTPromiseRejectBlock)reject) {
  NSMutableArray *features = [NSMutableArray new];
  BOOL fp16 = SysctlFlag("hw.optional.arm.FEAT_FP16") || SysctlFlag("hw.optional.neon_fp16");
  BOOL dot = SysctlFlag("hw.optional.arm.FEAT_DotProd");
  BOOL i8mm = SysctlFlag("hw.optional.arm.FEAT_I8MM");
  if (SysctlFlag("hw.optional.neon")) [features addObject:@"neon"];
  if (fp16) [features addObject:@"fp16"];
  if (dot) [features addObject:@"dotprod"];
  if (i8mm) [features addObject:@"i8mm"];
  if (SysctlFlag("hw.optional.arm.FEAT_BF16")) [features addObject:@"bf16"];
  NSString *machine = SysctlString("hw.machine");
  resolve(@{
    @"cores" : @([NSProcessInfo processInfo].activeProcessorCount),
    @"features" : features,
    @"hasFp16" : @(fp16),
    @"hasDotProd" : @(dot),
    @"hasSve" : @NO,
    @"hasI8mm" : @(i8mm),
    @"socModel" : SysctlString("machdep.cpu.brand_string"),
    @"hardware" : machine,
    @"maxFreqMhz" : @0,
#if TARGET_OS_SIMULATOR
    @"abis" : @[ @"simulator" ],
#else
    @"abis" : @[ @"arm64" ],
#endif
  });
}

RCT_EXPORT_METHOD(getGPUInfo : (RCTPromiseResolveBlock)resolve reject : (RCTPromiseRejectBlock)reject) {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  NSString *name = dev ? dev.name : @"";
  BOOL apple7 = NO;
  if (dev) apple7 = [dev supportsFamily:MTLGPUFamilyApple7];
  resolve(@{
    @"renderer" : name,
    @"vendor" : @"Apple",
    @"version" : apple7 ? @"Metal (Apple7+)" : @"Metal",
    @"hasAdreno" : @NO,
    @"hasMali" : @NO,
    @"hasPowerVR" : @NO,
    @"hasAppleGpu" : @(dev != nil),
    @"gpuType" : name.length ? name : @"Apple GPU",
  });
}

RCT_EXPORT_METHOD(getAvailableMemory : (RCTPromiseResolveBlock)resolve reject : (RCTPromiseRejectBlock)reject) {
  if (@available(iOS 13.0, *)) {
    resolve(@((double)os_proc_available_memory()));
  } else {
    resolve(@([NSProcessInfo processInfo].physicalMemory / 2.0));
  }
}

RCT_EXPORT_METHOD(hasNpu : (RCTPromiseResolveBlock)resolve reject : (RCTPromiseRejectBlock)reject) {
#if TARGET_OS_SIMULATOR
  resolve(@NO);
#else
  // Every A12+ chip has a Neural Engine; A12 is also the minimum for iOS 15+ on ANE-bearing SoCs.
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  resolve(@(dev != nil && [dev supportsFamily:MTLGPUFamilyApple5]));
#endif
}

#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:(const facebook::react::ObjCTurboModule::InitParams &)params {
  return std::make_shared<facebook::react::NativeHardwareInfoSpecJSI>(params);
}
#endif

@end

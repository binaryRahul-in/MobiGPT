#import <Foundation/Foundation.h>

#ifdef RCT_NEW_ARCH_ENABLED
#import <MobiGPTDeviceSpec/MobiGPTDeviceSpec.h>
@interface MobiGPTDevice : NSObject <NativeHardwareInfoSpec>
#else
#import <React/RCTBridgeModule.h>
@interface MobiGPTDevice : NSObject <RCTBridgeModule>
#endif
@end

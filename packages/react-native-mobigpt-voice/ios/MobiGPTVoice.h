#import <Foundation/Foundation.h>
#import <React/RCTBridgeModule.h>
#import <ReactCommon/RCTTurboModuleWithJSIBindings.h>

#ifdef RCT_NEW_ARCH_ENABLED
#import <MobiGPTVoiceSpec/MobiGPTVoiceSpec.h>
@interface MobiGPTVoice : NSObject <NativeMobiGPTVoiceSpec, RCTTurboModuleWithJSIBindings>
#else
@interface MobiGPTVoice : NSObject <RCTBridgeModule>
#endif
@end

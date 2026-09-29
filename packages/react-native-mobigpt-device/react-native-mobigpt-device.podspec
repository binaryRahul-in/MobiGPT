require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-mobigpt-device"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = "https://github.com/binaryRahul-in/MobiGPT"
  s.license      = package["license"]
  s.authors      = "MobiGPT contributors"
  s.platforms    = { :ios => "15.1" }
  s.source       = { :git => "https://github.com/binaryRahul-in/MobiGPT.git", :tag => "v#{s.version}" }
  s.source_files = "ios/**/*.{h,m,mm}"
  s.frameworks   = "Metal", "UIKit"
  install_modules_dependencies(s)
end

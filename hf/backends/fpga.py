import os


def gen_VCU440(system, FPGAFreq = 200):
    tmp_template = """
package chipyard.fpga.vcu440

import chipyard.DefaultClockFrequencyKey
import freechips.rocketchip.config.Config
import freechips.rocketchip.devices.tilelink.BootROMLocated
import freechips.rocketchip.diplomacy.DTSTimebase
import freechips.rocketchip.subsystem.ExtMem
import sifive.blocks.devices.spi.{PeripherySPIKey, SPIParams}
import sifive.blocks.devices.uart.{PeripheryUARTKey, UARTParams}
import sifive.fpgashells.shell.xilinx.{VCU440DDRSize, VCU440ShellPMOD}
import testchipip.SerialTLKey

import scala.sys.process._

class WithDefaultPeripherals extends Config((site, here, up) => {
  case PeripheryUARTKey => List(UARTParams(address = BigInt(0x64000000L)))
  case PeripherySPIKey => List(SPIParams(rAddress = BigInt(0x64001000L)))
  case VCU440ShellPMOD => "SDIO"
})

class WithSystemModifications extends Config((site, here, up) => {
  case DTSTimebase => BigInt((1e6).toLong)
  case BootROMLocated(x) => up(BootROMLocated(x), site).map { p =>
    // invoke makefile for sdboot
    val freqMHz = (site(DefaultClockFrequencyKey) * 1e6).toLong
    val make = s"make -C fpga/src/main/resources/vcu118/sdboot PBUS_CLK=${freqMHz} bin"
    require (make.! == 0, "Failed to build bootrom")
    p.copy(hang = 0x10000, contentFileName = s"./fpga/src/main/resources/vcu118/sdboot/build/sdboot.bin")
  }
  case ExtMem => up(ExtMem, site).map(x => x.copy(master = x.master.copy(size = site(VCU440DDRSize)))) // set extmem to DDR size
  case SerialTLKey => None // remove serialized tl port
})

// DOC include start: AbstractVCU440 and Rocket
class WithFPGAFrequency(fMHz: Double) extends Config(
  new chipyard.config.WithPeripheryBusFrequency(fMHz) ++ // assumes using PBUS as default freq.
  new chipyard.config.WithMemoryBusFrequency(fMHz)
)

class WithFPGAFreq25MHz extends WithFPGAFrequency(25)

class WithFPGAFreq50MHz extends WithFPGAFrequency(50)

class WithFPGAFreq75MHz extends WithFPGAFrequency(75)

class WithFPGAFreq100MHz extends WithFPGAFrequency(100)

class S2CConfigTweak extends Config(

  new WithFPGAFrequency(FPGAGLOBALFREQ) ++ // default 100MHz freq
  // harness binders
  new WithUART ++
  new WithSPISDCard ++
  new WithDDRMem ++
  // io binders
  new WithUARTIOPassthrough ++
  new WithSPIIOPassthrough ++
  new WithTLIOPassthrough ++
  // other configuration
  new WithDefaultPeripherals ++
  new chipyard.config.WithTLBackingMemory ++ // use TL backing memory
  new WithSystemModifications ++ // setup busses, use sdboot bootrom, setup ext. mem. size
  new chipyard.config.WithNoDebug ++ // remove debug module
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNMemoryChannels(1)
)

class CustomVCU440Config extends Config(
  new S2CConfigTweak ++
  new chipyard.BASESOCCONFIG
)
// DOC include end: AbstractVCU440 and Rocket
    """

    addLines = tmp_template.replace("FPGAGLOBALFREQ",str(FPGAFreq)).replace("BASESOCCONFIG",str(system.name))
    target = os.path.join(system.chipyard_path, 'fpga/src/main/scala/vcu440/Configs.scala')
    # backup
    with open(target, 'w') as tmpFile: #写入原文件
      print(addLines, file=tmpFile)

    os.chdir(os.path.join(system.chipyard_path, 'fpga'))
    os.system('make SUB_PROJECT=vcu440 bitstream')
    # os.system("cd "+fpgaRoot+"; make SUB_PROJECT=vcu440 bistream")
    # os.system("cp "+targetConfigFile+" "+targetConfigFile+"backup")

    '''
    with open (targetConfigFile, 'r') as tmpFile:#备份原文件
      backup = tmpFile.readlines()
    with open (targetConfigFile, 'w') as tmpFile:#恢复原文件
      print(backup, file=tmpFile)'''


if __name__ == "__main__":
  class Sys():
    name = 'RocketConfig'
    chipyard_path = '/home/ya-wang/Proj/chipyard'
  sys = Sys()
  gen_VCU440(sys)
  #os.system("cd chipyard/fpga/; make SUB_PROJECT=vcu440 bitstream")

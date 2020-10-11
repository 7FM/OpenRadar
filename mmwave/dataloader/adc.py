# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import codecs
import socket
import struct
from enum import Enum
import threading
import numpy as np
import time
from multiprocessing import Process, Queue


class CMD(Enum):
    RESET_FPGA_CMD_CODE = '0100'
    RESET_AR_DEV_CMD_CODE = '0200'
    CONFIG_FPGA_GEN_CMD_CODE = '0300'
    CONFIG_EEPROM_CMD_CODE = '0400'
    RECORD_START_CMD_CODE = '0500'
    RECORD_STOP_CMD_CODE = '0600'
    PLAYBACK_START_CMD_CODE = '0700'
    PLAYBACK_STOP_CMD_CODE = '0800'
    SYSTEM_CONNECT_CMD_CODE = '0900'
    SYSTEM_ERROR_CMD_CODE = '0a00'
    CONFIG_PACKET_DATA_CMD_CODE = '0b00'
    CONFIG_DATA_MODE_AR_DEV_CMD_CODE = '0c00'
    INIT_FPGA_PLAYBACK_CMD_CODE = '0d00'
    READ_FPGA_VERSION_CMD_CODE = '0e00'

    def __str__(self):
        return str(self.value)


# MESSAGE = codecs.decode(b'5aa509000000aaee', 'hex')
CONFIG_HEADER = '5aa5'
CONFIG_STATUS = '0000'
CONFIG_FOOTER = 'aaee'
# STATIC
MAX_PACKET_SIZE = 1514
BYTES_IN_PACKET = 1456
# BYTES_IN_PACKET = 1462


class AdcDataStreamer():
    def __init__(self, data_recv_cfg, bytes_in_frame, timeout=1, udp_raw_data=False, udp_raw_data_dir=None,
                 log_error_func=print, log_warning_func=print, log_info_func=print):
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # Bind data socket to fpga
        print("DCA IP:", data_recv_cfg)
        self.data_socket.bind(data_recv_cfg)
        self.timeout = timeout
        self.bytes_in_frame = bytes_in_frame
        # print("bytes in frame:", bytes_in_frame)
        self.uint16_in_frame = bytes_in_frame // 2
        self.uint16_in_packet = BYTES_IN_PACKET // 2
        self.startup = True
        self.running = True
        self.frame_byte_idx = 0
        self.backup_file = open(udp_raw_data_dir, "wb+") if udp_raw_data else None
        # self.ret_frame = np.zeros(self.uint16_in_frame, dtype=np.int16)
        # self.ret_frame = np.zeros(self.bytes_in_frame, dtype=bytes)
        # self.ret_frame = bytearray(self.bytes_in_frame)
        self.ret_frame = np.zeros(self.bytes_in_frame, dtype=np.uint8)
        self.last_byte_count = 0
        self.last_packet_num = 0
        self.last_frame_byte_idx = 0
        self.lost_packages = 0
        self.log_error_func = log_error_func
        self.log_warning_func = log_warning_func
        self.log_info_func = log_info_func

    def is_set_up(self):
        try:
            self.data_socket.settimeout(1)
            packet_num, byte_count, packet_data = self._read_data_packet()
            return True
        except socket.timeout as e:
            return False

    def stream(self, data_queue, time_queue):

        # while self.running:
        #     self.data_socket.settimeout(2)
        #     all_data = []
        #     print("Start")
        #     while True:
        #         try:
        #             p_num, b_count, data = self._read_data_packet()
        #             print(data)
        #             all_data.extend(data)
        #         except Exception as e:
        #             print(e)
        #             data_queue.put(all_data)
        #             self.running = False
        #             return all_data

        ret_frame = np.zeros(self.bytes_in_frame, dtype=np.uint8)
        time_read = time.time()
        
        while self.running:

            while self.startup:  # Wait for start of next frame
                self.data_socket.settimeout(self.timeout)
                try:
                    packet_num, byte_count, packet_data = self._read_data_packet()
                except Exception as e:
                    self.log_error_func(e)
                    # raise TimeoutError("Could not reveive from dca1000 using timeout (s):", self.timeout)  # TODO remove when timeout handled?
                self.last_byte_count = byte_count
                self.last_packet_num = packet_num
                if (byte_count + BYTES_IN_PACKET) % self.bytes_in_frame < BYTES_IN_PACKET:  # got first bytes of new frame
                    # self.frame_byte_idx = (byte_count % self.bytes_in_frame) // 2 or BYTES_IN_PACKET // 2  # old
                    self.frame_byte_idx = ((byte_count + BYTES_IN_PACKET) % self.bytes_in_frame) or len(packet_data)
                    # if self.frame_byte_idx > len(packet_data):
                    #     self.frame_byte_idx = len(packet_data)
                    # print("FrameByteIdx 1", self.frame_byte_idx)
                    # print("Range: ", 0, " ", self.frame_byte_idx)
                    # self.ret_frame[0:self.frame_byte_idx] = packet_data[-self.frame_byte_idx:]
                    ret_frame[0:self.frame_byte_idx] = packet_data[len(packet_data) - self.frame_byte_idx:]
                    # print(packet_data[len(packet_data) - self.frame_byte_idx:20])
                    self.startup = False
                    # break

            self.data_socket.settimeout(self.timeout)

            packet_num, byte_count, packet_data = self._read_data_packet()
            # print(packet_num, byte_count, len(packet_data))
            # new_byte_count = byte_count - self.last_byte_count
            # new_byte_idx = new_byte_count // 2
            new_byte_idx = len(packet_data)
            # print(new_byte_idx)

            if self.last_packet_num + 1 == packet_num:  # check if packets are being dropped
                # if self.frame_byte_idx + new_byte_idx >= self.uint16_in_frame:  # old
                # print(self.frame_byte_idx + new_byte_idx, self.bytes_in_frame)
                if self.frame_byte_idx + new_byte_idx >= self.bytes_in_frame:  # got a whole frame, put it in queue
                    # overshoot_idx = self.frame_byte_idx + new_byte_idx - self.uint16_in_frame  # old
                    # print("FrameByteIdx 2", self.frame_byte_idx)
                    # print("Range: ", self.frame_byte_idx, " end")
                    ret_frame[self.frame_byte_idx:] = \
                        packet_data[:self.bytes_in_frame - self.frame_byte_idx]
                        # packet_data[:self.uint16_in_frame - self.frame_byte_idx]  # old
                    data_queue.put(ret_frame.tobytes())
                    time_queue.put(time_read)
                    added_bytes = self.bytes_in_frame - self.frame_byte_idx
                    self.frame_byte_idx = new_byte_idx - added_bytes
                    # to_add = new_byte_idx - self.frame_byte_idx
                    # overshoot_idx = self.frame_byte_idx + new_byte_idx - self.bytes_in_frame
                    # self.frame_byte_idx = new_byte_idx - overshoot_idx  # new_byte_count - (self.bytes_in_frame // 2 - self.frame_byte_idx)
                    # print("FrameByteIdx 3", self.frame_byte_idx)
                    # print("Range: ", 0, " ", self.frame_byte_idx)
                    if self.frame_byte_idx > 0:
                        ret_frame[:self.frame_byte_idx] = packet_data[-self.frame_byte_idx:]
                    # else:
                    #     self.frame_byte_idx = BYTES_IN_PACKET
                    time_read = time.time()
                else:
                    # print("FrameByteIdx 4", self.frame_byte_idx)
                    # print("Range: ", self.frame_byte_idx, " ", self.frame_byte_idx + new_byte_idx)
                    ret_frame[self.frame_byte_idx:self.frame_byte_idx + new_byte_idx] = packet_data
                    self.frame_byte_idx += new_byte_idx
            else:
                self.lost_packages += 1
                self.log_warning_func("Lost package count: {}".format(self.lost_packages))
                self.startup = True

            self.last_byte_count = byte_count
            self.last_packet_num = packet_num
            self.last_frame_byte_idx= self.frame_byte_idx

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)  # TODO handle timeouts?
        packet_num = struct.unpack('<1l', data[:4])[0]
        # print("packet_num", packet_num)
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        # packet_data = data[10:]
        packet_data = np.frombuffer(data[10:], dtype=np.uint8)
        if self.backup_file is not None:  # backup raw UDP traffic
            self.backup_file.write(data)
        # packet_data = np.frombuffer(data[10:], dtype=np.int16)
        return packet_num, byte_count, packet_data

    def close(self):
        self.data_socket.close()

class DCA1000:
    """Software interface to the DCA1000 EVM board via ethernet.

    Attributes:
        static_ip (str): IP to receive data from the FPGA
        adc_ip (str): IP to send configuration commands to the FPGA
        data_port (int): Port that the FPGA is using to send data
        config_port (int): Port that the FPGA is using to read configuration commands from


    General steps are as follows:
        1. Power cycle DCA1000 and XWR1xxx sensor
        2. Open mmWaveStudio and setup normally until tab SensorConfig or use lua script
        3. Make sure to connect mmWaveStudio to the board via ethernet
        4. Start streaming data
        5. Read in frames using class

    Examples:
        >>> dca = DCA1000()
        >>> adc_data = dca.read(timeout=.1)
        >>> frame = dca.organize(adc_data, 128, 4, 256)

    """

    def __init__(self, static_ip='192.168.33.30', adc_ip='192.168.33.180', timeout=1,
                 data_port=4098, config_port=4096, num_loops_per_frame=16, num_rx=4, num_tx=3, num_adc_samples=240,
                 udp_raw_data=0, udp_raw_data_dir=None,
                 log_error_func=print, log_warning_func=print, log_info_func=print):
        # Save network data
        # self.static_ip = static_ip
        # self.adc_ip = adc_ip
        # self.data_port = data_port
        # self.config_port = config_port

        adc_params = {'chirps': 16,  # 32 TODO chirps
                      'rx': num_rx,
                      'tx': num_tx,
                      'samples': num_adc_samples,
                      'IQ': 2,
                      'bytes': 2}
        # DYNAMIC
        self.bytes_in_frame = (adc_params['chirps'] * adc_params['rx'] * adc_params['tx'] *
                               adc_params['IQ'] * adc_params['samples'] * adc_params['bytes'])
        self.bytes_in_frame_clipped = (self.bytes_in_frame // BYTES_IN_PACKET) * BYTES_IN_PACKET
        self.packets_in_frame = self.bytes_in_frame / BYTES_IN_PACKET
        self.packets_in_frame_clipped = self.bytes_in_frame // BYTES_IN_PACKET
        self.uint16_in_packet = BYTES_IN_PACKET // 2
        self.uint16_in_frame = self.bytes_in_frame // 2

        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        # self.data_socket = socket.socket(socket.AF_INET,
        #                                  socket.SOCK_DGRAM,
        #                                  socket.IPPROTO_UDP)

        # Bind data socket to fpga
        # self.data_socket.bind(self.data_recv)

        self.adc_data_streamer = AdcDataStreamer(self.data_recv, self.bytes_in_frame, timeout=timeout, 
                                                 udp_raw_data=udp_raw_data, udp_raw_data_dir=udp_raw_data_dir,
                                                 log_error_func=log_error_func, log_warning_func=log_warning_func,
                                                 log_info_func=log_info_func)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)

        self.data = []
        self.packet_count = []
        self.byte_count = []

        self.frame_buff = []

        self.curr_buff = None
        self.last_frame = None

        self.lost_packets = None
        self.producer = None
        self.data_queue = None
        self.time_queue = None

    def start_streaming(self, data_queue, time_queue=Queue(30)):
        self.data_queue = data_queue
        self.time_queue = time_queue
        self.producer = Process(target=self.adc_data_streamer.stream, args=(data_queue, time_queue,))
        self.producer.start()

    def read_adc(self):
        return self.data_queue.get(), self.time_queue.get()

    def configure(self):
        """Initializes and connects to the FPGA

        Returns:
            None

        """
        # SYSTEM_CONNECT_CMD_CODE
        # 5a a5 09 00 00 00 aa ee
        print(self._send_command(CMD.SYSTEM_CONNECT_CMD_CODE))

        # READ_FPGA_VERSION_CMD_CODE
        # 5a a5 0e 00 00 00 aa ee
        print(self._send_command(CMD.READ_FPGA_VERSION_CMD_CODE))

        # CONFIG_FPGA_GEN_CMD_CODE
        # 5a a5 03 00 06 00 01 02 01 02 03 1e aa ee
        print(self._send_command(CMD.CONFIG_FPGA_GEN_CMD_CODE, '0600', 'c005350c0000'))

        # CONFIG_PACKET_DATA_CMD_CODE 
        # 5a a5 0b 00 06 00 c0 05 35 0c 00 00 aa ee
        print(self._send_command(CMD.CONFIG_PACKET_DATA_CMD_CODE, '0600', 'c005350c0000'))

    def close(self):
        """Closes the sockets that are used for receiving and sending data

        Returns:
            None

        """
        self.adc_data_streamer.data_socket.close()
        self.config_socket.close()

    def read(self, timeout=30):
        """ Read in a single packet via UDP

        Args:
            timeout (float): Time to wait for packet before moving on

        Returns:
            Full frame as array if successful, else None

        """
        # Configure
        self.data_socket.settimeout(timeout)

        # Frame buffer
        # ret_frame = np.zeros(UINT16_IN_FRAME, dtype=np.uint16)
        ret_frame = np.zeros(self.uint16_in_frame, dtype=np.int16)

        frame_byte_idx = 0

        packets_read = 1
        # Wait for start of next frame
        before = 0
        while True:
            packet_num, byte_count, packet_data = self._read_data_packet()
            # print(BYTES_IN_FRAME_CLIPPED, " : ", byte_count, " ; ", byte_count % BYTES_IN_FRAME_CLIPPED,
            #       " ... ", byte_count - before)
            # before = byte_count
            # if byte_count % self.bytes_in_frame_clipped == 0:
            # if byte_count % self.bytes_in_frame < BYTES_IN_PACKET:
            if byte_count % self.bytes_in_frame < 1514:
                # frame_byte_idx = byte_count % self.bytes_in_frame
                frame_byte_idx = byte_count % 1500
                ret_frame[0:frame_byte_idx] = packet_data[-frame_byte_idx:]
                break
            # if byte_count % BYTES_IN_FRAME == 0:
            #     packets_read = 1
            #     ret_frame[0:self.uint16_in_packet] = packet_data
            #     break

        # Read in the rest of the frame            
        while True:
            # print("Wait for rest of frame")
            packet_num, byte_count, packet_data = self._read_data_packet()
            packets_read += 1

            if byte_count % self.bytes_in_frame_clipped == 0:
                self.lost_packets = self.packets_in_frame_clipped - packets_read
                return ret_frame

            curr_idx = ((packet_num - 1) % self.packets_in_frame_clipped)
            try:
                ret_frame[curr_idx * self.uint16_in_packet:(curr_idx + 1) * self.uint16_in_packet] = packet_data
            except:
                pass

            if packets_read > self.packets_in_frame_clipped:
                packets_read = 0

    def _send_command(self, cmd, length='0000', body='', timeout=1):
        """Helper function to send a single commmand to the FPGA

        Args:
            cmd (CMD): Command code to send to the FPGA
            length (str): Length of the body of the command (if any)
            body (str): Body information of the command
            timeout (int): Time in seconds to wait for socket data until timeout

        Returns:
            str: Response message

        """
        # Create timeout exception
        self.config_socket.settimeout(timeout)

        # Create and send message
        resp = ''
        msg = codecs.decode(''.join((CONFIG_HEADER, str(cmd), length, body, CONFIG_FOOTER)), 'hex')
        try:
            self.config_socket.sendto(msg, self.cfg_dest)
            resp, addr = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        except socket.timeout as e:
            print(e)
        return resp

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.int16)  # TODO
        return packet_num, byte_count, packet_data

    def _listen_for_error(self):
        """Helper function to try and read in for an error message from the FPGA

        Returns:
            None

        """
        self.config_socket.settimeout(None)
        msg = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        if msg == b'5aa50a000300aaee':
            print('stopped:', msg)

    def _stop_stream(self):
        """Helper function to send the stop command to the FPGA

        Returns:
            str: Response Message

        """
        return self._send_command(CMD.RECORD_STOP_CMD_CODE)

    @staticmethod
    def organize(raw_frame, num_chirps, num_rx, num_samples, stream=False):
        """Reorganizes raw ADC data into a full frame

        Args:
            raw_frame (ndarray): Data to format
            num_chirps: Number of chirps included in the frame
            num_rx: Number of receivers used in the frame
            num_samples: Number of ADC samples included in each chirp

        Returns:
            ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)
            chirps are sorted as follows:
                -e.g. one tx antenna at a time, 3 in total: [txA, txB, txC] == [tx0, tx2, tx1]
                -e.g. multiple tx antenna at a time, 3 in total: [txA, txB, txC] == [[tx0, tx1], tx2, tx1],
                    order as per chirp config order send to xwr1xxx
                -N = num_tx * num_chirps
                -chirps = [ chirp1_txA, chirp1_txB, chirp1_txC,
                            chirp2_txA, chirp2_txB, chirp2_txC,
                            chirpN_txA, chirpN_txB, chirpN_txC ]
                -e.g 3 TX Antennas (num_tx == 3), 16 chirps per Frame:
                    - Antennas and Chirps with 1 indexing (e.g tx1, tx2, tx3; Chirp1, Chirp2, ..., Chirp16):
                        chirps[28] == Chirp10 & num_tx2 == (10 - 1) * num_tx + (2 - 1)
                        chirps[X] == Chirp(X // 3 + 1) + num_tx(X mod 3 + 1)
                    - Antennas and Chirps with 0 indexing (e.g tx0, tx1, tx2; Chirp0, Chirp1, ..., Chirp15):
                        chirps[28] == Chirp9 & num_tx1 == 9 * num_tx + 1
                        chirps[X] == Chirp(X // 3) + num_tx(X mod 3)

        """
        ret = np.zeros(len(raw_frame) // 2, dtype=complex)
        # TODO reshape depending on antenna parameter etc
        if stream:
            # raw_frame = np.dstack((raw_frame[:92160//2], raw_frame[92160//2:])).flatten()
            ret[0::4] = raw_frame[0::8] + 1j * raw_frame[4::8]
            ret[1::4] = raw_frame[1::8] + 1j * raw_frame[5::8]
            ret[2::4] = raw_frame[2::8] + 1j * raw_frame[6::8]
            ret[3::4] = raw_frame[3::8] + 1j * raw_frame[7::8]
            ret = ret.reshape((num_chirps, num_samples, num_rx))
            ret = ret.transpose((0, 2, 1))
        else:
            # Separate IQ data
            ret[0::4] = raw_frame[0::8] + 1j * raw_frame[4::8]
            ret[1::4] = raw_frame[1::8] + 1j * raw_frame[5::8]
            ret[2::4] = raw_frame[2::8] + 1j * raw_frame[6::8]
            ret[3::4] = raw_frame[3::8] + 1j * raw_frame[7::8]
            # ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
            # ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
            ret = ret.reshape((num_chirps, num_samples, num_rx))
            ret = ret.transpose((0, 2, 1))
        return ret

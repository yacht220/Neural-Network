<logheader platform="Linux" env-locale-cs="ANSI_X3.4-1968" process_id="MOD-1510822024625" version="7.5.100-183" tz="EST" />

--------------------------------------------------------------------------------------------------------
-----  START MOD-1510822024625 log 2017-11-16 03:47:48 EST  [7.5.100-183 Linux-x86_64]
--------------------------------------------------------------------------------------------------------

2017-11-16 03:47:48 MOD-1510822024625 Info <0000>: Workorder : MOD-1510822024625
2017-11-16 03:47:48 MOD-1510822024625 Info <0000>: Plugin ID : 14003-Isilon Filer
2017-11-16 03:47:48 MOD-1510822024625 Info <0000>: bindir    : /usr/local/avamar/bin
2017-11-16 03:47:48 MOD-1510822024625 Info <0000>: vardir    : /usr/local/avamar/var/clidev-isilon2-2-vm-griffin-6173
2017-11-16 03:47:48 MOD-1510822024625 Info <0000>: Initiator : 
2017-11-16 03:47:48 MOD-1510822024625 Info <0000>: Order      : 
<restore pid="Isilon Filer" time="1510822045" ack="false" type="work" msgver="5" pidnum="14003" work="restore" ticket="XXX" key="14003-MOD-1510822024625" sync="bg" sessionid="bd5f0db7f2a0d5dcae698290df963531b2541d0c" cid="0a4fc2dcc3ff974d81f16ce9c595fa5162d0d1b5" mid="3844/3863/41" gsansid="9151082202477009" last_ctl_man_key="14003-MOD-1510822024625" wid="MOD-1510822024625" customaction="">
  <targetlist>
    <path name="/ifs/100MB_br/">
      <dir inode="1080424314" date="2017-11-16 00:45:21" internal="0" group="0" protection="drwxrwx---" name="/ifs/100MB_br" size="555942876" links="8" user="0" />
    </path>
  </targetlist>
  <directives>
    <flag type="string" value="tls" name="encrypt" />
    <flag type="string" value="high" name="encrypt-strength" />
    <flag type="string" value="7" name="labelnum" />
    <flag type="string" value="/yizhou/clidev-isilon2-2-vm-griffin-6173" name="path" />
    <flag type="string" value="vm-griffin-6177.asl.lab.emc.com" name="hfsaddr" />
    <flag type="string" value="27000" name="hfsport" />
    <flag type="password" value="****" name="ddr-auth-token-key" />
    <flag type="string" value="restoreonly" name="id" />
    <flag type="password" value="****" name="ap" />
  </directives>
  <catalog>
    <flag type="pulldown" value="default" name="[avtar]ddr-encrypt-strength" />
    <flag type="pulldown" value="1" name="ddr-index" />
    <flag type="pulldown" pidnum="14003" value="4" name="max-streams" />
  </catalog>
  <browse-from-catalog />
  <free-form />
</restore>

2017-11-16 03:47:48 MOD-1510822024625 Info <0000>: Executable : /usr/local/avamar/bin/avndmp
2017-11-16 03:47:48 MOD-1510822024625 Info <0000>: Args       : avndmp --sysdir="/usr/local/avamar/etc/clidev-isilon2-2-vm-griffin-6173" --bindir="/usr/local/avamar/bin" --vardir="/usr/local/avamar/var/clidev-isilon2-2-vm-griffin-6173" --ctlcallport="41737" --ctlinterface="14003-MOD-1510822024625" --logtimeformat="default" --logtimezone="default" --logfile="/usr/local/avamar/var/clidev-isilon2-2-vm-griffin-6173/MOD-1510822024625-14003-Isilon_Filer.log"
<SUMMARY totalprotectedbytes="556225024" streambytes="558322176" filebytes_ddsent="556194304" filecount_excluded="0" dirbytes_sent="29664" filebytes_sent="9079" compbytes_ispresent="0" filebytes_ispresent="0" dirbytes_hashcache="0" filebytes_skipped="0" totalprimarybytes="555942876" compbytes_sent="0" dirbytes_ispresent="0" chunkcount_sent="15" chunkcount_hashcache="0" filebytes_hardlink="0" dircount_sent="15" filebytes_noreason="0" filebytes_reduced="21641" filecount_hardlink="0" filecount_noreason="0" filecount_filecache="0" filebytes_ddpresent="0" filecount_skipped="0" compcount_hashcache="0" dircount_hashcache="0" snapview_stat="false" filebytes_filecache="0" compcount_sent="0" dircount_ispresent="0" chunkcount_ispresent="0" elapsed="11" compbytes_hashcache="0" filebytes_hashcache="0" filecount_reduced="15" compcount_ispresent="0" totalfilebytes="556225024" totalfilecount="110" filebytes_excluded="0"><session numthreads="1" type="avtarrestore" expires="0" ndispatchers="1" domain="" workorderid="MOD-1510822024625" numconns="1" pidnum="14003" label="MOD-1510821908238#0" path="/yizhou/clidev-isilon2-2-vm-griffin-6173" labelnum="7" encrypt="tls" starttime="1510822073" dispatcher0="10.25.106.177:29000" sessionid="9151082202477009" root="" pluginid="Isilon Filer" encrypt-strength="high" user="" clientid="0a4fc2dcc3ff974d81f16ce9c595fa5162d0d1b5"><host numprocs="1" speed="200" osuser="root" name="vm-griffin-6173" memory="934" /><build msgversion="13-10" time="11:47:51" appname="avtar" zlibversion="1.2.3" lzoversion="1.08 Jul 12 2002" date="Jun 12 2017" appversion="7.5.100-183" processortype="x86_64" osversion="SLES-64" sslversion="TLSv1   OpenSSL 1.0.2a-fips 19 Mar 2015" osname="Linux" /><dirstats /></session><errorsummary exitcode="0" errors="0" warnings="0" fatals="0" /></SUMMARY>
<SUMMARY totalprotectedbytes="556225024" streambytes="558322176" filebytes_ddsent="556194304" filecount_excluded="0" dirbytes_sent="29664" filebytes_sent="9079" compbytes_ispresent="0" filebytes_ispresent="0" dirbytes_hashcache="0" filebytes_skipped="0" totalprimarybytes="555942876" compbytes_sent="0" dirbytes_ispresent="0" chunkcount_sent="15" chunkcount_hashcache="0" filebytes_hardlink="0" dircount_sent="15" filebytes_noreason="0" filebytes_reduced="21641" filecount_hardlink="0" filecount_noreason="0" filecount_filecache="0" filebytes_ddpresent="0" filecount_skipped="0" compcount_hashcache="0" dircount_hashcache="0" snapview_stat="false" filebytes_filecache="0" compcount_sent="0" dircount_ispresent="0" chunkcount_ispresent="0" elapsed="11" compbytes_hashcache="0" filebytes_hashcache="0" filecount_reduced="15" compcount_ispresent="0" totalfilebytes="556225024" totalfilecount="110" filebytes_excluded="0">
  <session numthreads="1" type="avtarrestore" expires="0" ndispatchers="1" domain="" workorderid="MOD-1510822024625" numconns="1" pidnum="14003" label="MOD-1510821908238#0" path="/yizhou/clidev-isilon2-2-vm-griffin-6173" labelnum="7" encrypt="tls" starttime="1510822073" dispatcher0="10.25.106.177:29000" sessionid="9151082202477009" root="" pluginid="Isilon Filer" encrypt-strength="high" user="" clientid="0a4fc2dcc3ff974d81f16ce9c595fa5162d0d1b5">
    <host numprocs="1" speed="200" osuser="root" name="vm-griffin-6173" memory="934" />
    <build msgversion="13-10" time="11:47:51" appname="avtar" zlibversion="1.2.3" lzoversion="1.08 Jul 12 2002" date="Jun 12 2017" appversion="7.5.100-183" processortype="x86_64" osversion="SLES-64" sslversion="TLSv1   OpenSSL 1.0.2a-fips 19 Mar 2015" osname="Linux" />
    <dirstats />
  </session>
  <errorsummary exitcode="0" errors="0" warnings="0" fatals="0" />
</SUMMARY>

2017-11-16 03:48:09 MOD-1510822024625 Info <0000>: <workorder_finished pid="Isilon Filer" startUTC="2017-11-16 08:47:48" msgver="1" pidnum="14003" finishUTC="2017-11-16 08:48:09" errorcode="0" wid="MOD-1510822024625" errormessage="code 0: success" errorrealm="Plugin 14003-Isilon Filer" />

